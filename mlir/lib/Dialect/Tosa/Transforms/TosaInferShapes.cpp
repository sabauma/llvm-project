//===- TosaInferShapes.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Propogate shapes forward along TOSA operations to resolve dynamic shape
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAINFERSHAPES
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

// Check whether this use case is inferable (i.e. its type may be updated by
// shape propagation). We define an op as being inferable if it is used by a
// TosaOp, or an op with a type-inference related interface. When a
// non-replaceable use is encountered, the value is wrapped in a cast back to
// the original type after inference.
//
bool isInferableOperation(Operation *user, bool aggressiveMode) {
  return user->getDialect()->getNamespace() == TosaDialect::getDialectNamespace() ||
         isa<InferShapedTypeOpInterface>(user) ||
         (aggressiveMode && isa<InferTypeOpInterface>(user));
}

struct TypeRewriteState {
  void recordUpdate(Value value, Type oldType) {
    for (auto &user : value.getUses()) {
      if (!isInferableOperation(user.getOwner(), aggressiveMode))
        replacedUses.push_back({&user, oldType});
    }
  }

  void applyUpdates(RewriterBase& rewriter) {
    for (auto [operand, oldType] : replacedUses) {
      rewriter.setInsertionPoint(operand->getOwner());

      auto oldValue = operand->get();

      auto loc = oldValue.getLoc();
      auto castOp = rewriter.create<tensor::CastOp>(loc, oldType, oldValue);
      operand->set(castOp);
    }
  }

  bool aggressiveMode;

  using Entry = std::pair<OpOperand*, Type>;
  llvm::SmallVector<Entry> replacedUses = {};
};

struct ShapePropagation {
  void propagateShapesToTosaIf(Operation &op);
  void propagateShapesToTosaWhile(Operation &op);
  void propagateShapesInterface(InferShapedTypeOpInterface shapeInterface,
                                TypeRewriteState &requiresUpdate);

  void propagateShapesInterface(InferTypeOpInterface shapeInterface,
                                TypeRewriteState &requiresUpdate);

  void propagateShapesInRegion(Region &region);

  bool aggressiveMode{false};
};

void ShapePropagation::propagateShapesToTosaIf(Operation &op) {
  IfOp ifOp = dyn_cast<IfOp>(op);
  if (!ifOp)
    return;

  for (auto &region : op.getRegions()) {
    Block &frontBlock = region.front();
    if (frontBlock.getNumArguments() + 1 != ifOp.getNumOperands())
      return;

    for (unsigned int i = 1, s = op.getNumOperands(); i < s; i++) {
      auto inferredTy = cast<ShapedType>(op.getOperand(i).getType());
      auto blockArg = frontBlock.getArgument(i - 1);
      auto oldType = cast<ShapedType>(blockArg.getType());

      if (inferredTy.hasRank()) {
        Type newType = oldType.clone(inferredTy.getShape());
        blockArg.setType(newType);
      }
    }

    for (int i = 0, e = frontBlock.getNumArguments(); i < e; i++) {
      ValueKnowledge operandKnowledge = ValueKnowledge::getKnowledgeFromType(
          ifOp.getOperand(i + 1).getType());
      ValueKnowledge blockKnowledge = ValueKnowledge::getKnowledgeFromType(
          frontBlock.getArgument(i).getType());
      ValueKnowledge joinedKnowledge =
          ValueKnowledge::join(operandKnowledge, blockKnowledge);
      if (!joinedKnowledge)
        continue;
      frontBlock.getArgument(i).setType(joinedKnowledge.getType());
    }

    propagateShapesInRegion(region);
  }
}

void ShapePropagation::propagateShapesToTosaWhile(Operation &op) {
  WhileOp whileOp = dyn_cast<WhileOp>(op);
  if (!whileOp)
    return;

  // Determine what the expected argument types are to the cond/body blocks.
  // The expected arguments should be compatible with ever iteration of the
  // loop body / condition for tosa.while.
  llvm::SmallVector<Type> argTypes;
  for (auto operand : op.getOperands()) {
    auto operandTy = cast<ShapedType>(operand.getType());
    if (operandTy.hasRank()) {
      auto newTy = operandTy.clone(operandTy.getShape());
      argTypes.push_back(newTy);
    } else {
      argTypes.push_back(operand.getType());
    }
  }

  // Save out the type information so we can restore at the end.
  llvm::DenseMap<Value, Type> originalTypeMap;
  for (auto &block : op.getRegion(1)) {
    for (auto arg : block.getArguments())
      originalTypeMap[arg] = arg.getType();
    for (auto &op : block)
      for (auto result : op.getResults())
        originalTypeMap[result] = result.getType();
  }

  bool hasNewTypes = true;
  while (hasNewTypes) {

    // Set types on the block args.
    Region &bodyRegion = op.getRegion(1);
    Block &block = bodyRegion.front();
    for (int i = 0, s = argTypes.size(); i < s; i++) {
      block.getArgument(i).setType(argTypes[i]);
    }

    // Propagate to the end.
    propagateShapesInRegion(bodyRegion);

    // Find all the tosa yield types and verify there is atleast one.
    llvm::SmallVector<YieldOp> yieldOps;
    for (auto &block : bodyRegion)
      if (auto yieldOp = dyn_cast<YieldOp>(block.getTerminator()))
        yieldOps.push_back(yieldOp);

    if (yieldOps.empty())
      return;

    // Using the new tosa.yield operand types, infer the new subtypes.
    llvm::SmallVector<ValueKnowledge> yieldTypeInfo;
    for (auto ty : argTypes) {
      yieldTypeInfo.push_back(ValueKnowledge::getKnowledgeFromType(ty));
    }

    for (auto yieldOp : yieldOps) {
      for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
        auto newKnowledge =
            ValueKnowledge::getKnowledgeFromType(it.value().getType());
        yieldTypeInfo[it.index()] =
            ValueKnowledge::meet(yieldTypeInfo[it.index()], newKnowledge);
      }
    }

    // This should never happen.
    if (yieldTypeInfo.size() != argTypes.size()) {
      op.emitWarning("has a tosa.yield with the incorrect number of operands");
      return;
    }

    // Determine the new block args and see if any changed.
    hasNewTypes = false;
    for (int i = 0, s = yieldTypeInfo.size(); i < s; i++) {
      Type newType = yieldTypeInfo[i].getType();
      hasNewTypes |= (newType != argTypes[i]);
      argTypes[i] = newType;
    }

    // The types inferred in the block assume the operand types specified for
    // this iteration. We need to restore the original types to ensure that
    // future iterations only use the already specified types, not possible
    // types from previous iterations.
    for (auto &block : bodyRegion) {
      for (auto arg : block.getArguments())
        arg.setType(originalTypeMap[arg]);
      for (auto &op : block)
        for (auto result : op.getResults())
          result.setType(originalTypeMap[result]);
    }
  }

  // We now set the block arguments according to the most recent shape
  // inference results. This gives us the block arg types for the next
  // iteration.
  for (auto &region : op.getRegions()) {
    for (unsigned int i = 0, s = argTypes.size(); i < s; i++) {
      region.front().getArgument(i).setType(argTypes[i]);
    }

    propagateShapesInRegion(region);
  }
}

void ShapePropagation::propagateShapesInterface(InferShapedTypeOpInterface shapeInterface,
                                                TypeRewriteState &requiresUpdate) {
  Operation &op = *shapeInterface.getOperation();

  SmallVector<ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(
              op.getContext(), op.getLoc(), op.getOperands(),
              op.getDiscardableAttrDictionary(), op.getPropertiesStorage(),
              op.getRegions(), returnedShapes)
          .failed())
    return;

  for (auto it : llvm::zip(op.getResults(), returnedShapes)) {
    Value result = std::get<0>(it);
    ShapedTypeComponents predictedShape = std::get<1>(it);

    // Determine the knowledge based on the output type.
    // TODO: should also query WIP type probably
    Type resultTy = result.getType();
    auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(resultTy);

    // Compute the knowledge based on the inferred type.
    auto inferredKnowledge =
        ValueKnowledge(predictedShape.hasRank(),
                       predictedShape.hasRank() ? predictedShape.getDims()
                                                : ArrayRef<int64_t>{},
                       cast<ShapedType>(resultTy).getElementType());

    // Compute the new type based on the joined version.
    auto newKnowledge =
        ValueKnowledge::join(currentKnowledge, inferredKnowledge);
    if (!newKnowledge)
      continue;

    // Set new type
    result.setType(newKnowledge.getType());

    // Collect all uses of the operation which require update.
    requiresUpdate.recordUpdate(result, resultTy);
  }
}

void ShapePropagation::propagateShapesInterface(InferTypeOpInterface shapeInterface,
                                                TypeRewriteState &requiresUpdate) {
  Operation &op = *shapeInterface.getOperation();

  llvm::SmallVector<Type> returnedTypes;

  if (failed(shapeInterface.inferReturnTypes(
          op.getContext(), op.getLoc(), op.getOperands(),
          op.getDiscardableAttrDictionary(), op.getPropertiesStorage(),
          op.getRegions(), returnedTypes))) {
    return;
  }

  for (auto [result, predicted] : llvm::zip(op.getResults(), returnedTypes)) {
    auto resultTy = result.getType();

    if (resultTy == predicted || !isa<ShapedType>(result.getType()) ||
        !isa<ShapedType>(predicted))
      continue;

    // Compute the new type.
    auto newKnowledge =
        ValueKnowledge::join(ValueKnowledge::getKnowledgeFromType(predicted),
                             ValueKnowledge::getKnowledgeFromType(resultTy));
    if (!newKnowledge)
      continue;

    result.setType(newKnowledge.getType());

    requiresUpdate.recordUpdate(result, resultTy);
  }
}


void ShapePropagation::propagateShapesInRegion(Region &region) {

  TypeRewriteState requiresUpdate{aggressiveMode};
  for (auto &block : region) {
    for (Operation &op : block) {
      if (!isInferableOperation(&op, aggressiveMode))
        continue;

      propagateShapesToTosaIf(op);
      propagateShapesToTosaWhile(op);

      if (auto shapeInterface = dyn_cast<InferShapedTypeOpInterface>(op))
        propagateShapesInterface(shapeInterface, requiresUpdate);

      if (auto shapeInterface = dyn_cast<InferTypeOpInterface>(op))
        propagateShapesInterface(shapeInterface, requiresUpdate);
    }
  }

  // For each use whose type changed, cast the value with the new type back to
  // the old type.
  IRRewriter rewriter(region.getContext());
  requiresUpdate.applyUpdates(rewriter);
}

/// Pass that performs shape propagation across TOSA operations. This includes
/// migrating to within the regions of if/while operations.
struct TosaInferShapes
    : public tosa::impl::TosaInferShapesBase<TosaInferShapes> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    ShapePropagation propagation{aggressiveMode};
    propagation.propagateShapesInRegion(func.getBody());
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaInferShapesPass() {
  return std::make_unique<TosaInferShapes>();
}

std::unique_ptr<Pass> mlir::tosa::createTosaInferShapesPass(const TosaInferShapesOptions& options) {
  return std::make_unique<TosaInferShapes>(options);
}
