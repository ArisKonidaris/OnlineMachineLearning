package mlAPI.mlParameterServers

import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.math.DenseVector
import mlAPI.parameters.ParameterDescriptor

/**
 * An abstract base class of a Machine Learning Parameter Server that keeps the global model in a flat vector.
 *
 * @tparam WorkerIfc The remote interface of the Machine Learning worker.
 * @tparam QueryIfc  The remote interface of the querier.
 */
abstract class VectoredPS[WorkerIfc, QueryIfc] extends MLParameterServer[WorkerIfc, QueryIfc] {

  var globalModel: BreezeDenseVector[Double] = _

  def deserializeVector(remoteModelDescriptor: ParameterDescriptor): BreezeDenseVector[Double] =
    BreezeDenseVector(remoteModelDescriptor.getParams.asInstanceOf[DenseVector].data)

  def assertWarmup(modelDescriptor: ParameterDescriptor): Unit =
    assert(getCurrentCaller == 0 && modelDescriptor.getFitted > 0 && globalModel == null)

}
