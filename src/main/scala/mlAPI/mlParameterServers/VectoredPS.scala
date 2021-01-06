package mlAPI.mlParameterServers

import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector}
import mlAPI.math.{DenseVector, SparseVector}
import mlAPI.parameters.ParameterDescriptor

/**
 * An abstract base class of a Machine Learning Parameter Server that keeps the global model in a flat vector.
 *
 * @tparam WorkerIfc The remote interface of the Machine Learning worker.
 * @tparam QueryIfc  The remote interface of the querier.
 */
abstract class VectoredPS[WorkerIfc, QueryIfc] extends MLParameterServer[WorkerIfc, QueryIfc] {

  var globalModel: BreezeDenseVector[Double] = _

  def deserializeVector(remoteModelDescriptor: ParameterDescriptor): BreezeDenseVector[Double] = {
    remoteModelDescriptor.getParams match {
      case dense: DenseVector => BreezeDenseVector(dense.data)
      case sparse: SparseVector => BreezeDenseVector(sparse.toDenseVector.data)
      case _: Throwable => throw new RuntimeException("Unknown Vector model.")
    }
  }

  def assertWarmup(modelDescriptor: ParameterDescriptor): Unit =
    assert(getCurrentCaller == 0 && modelDescriptor.getFitted > 0 && globalModel == null)

}
