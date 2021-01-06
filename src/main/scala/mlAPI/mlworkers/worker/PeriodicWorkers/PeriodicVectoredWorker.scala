package mlAPI.mlworkers.worker.PeriodicWorkers

import mlAPI.math.{DenseVector, SparseVector}
import mlAPI.mlworkers.worker.VectoredWorker
import mlAPI.parameters.ParameterDescriptor

/** An abstract base class of a periodic Online Machine Learning worker with a vectored model.
 *
 * @tparam ProxyIfc The remote interface of the Parameter Server.
 * @tparam QueryIfc The remote interface of the querier.
 */
abstract class PeriodicVectoredWorker[ProxyIfc, QueryIfc]() extends VectoredWorker[ProxyIfc, QueryIfc] {

  /** A method called each type the new global model
   * (or a slice of it) arrives from the parameter server.
   */
  def updateModel(mDesc: ParameterDescriptor): Unit = {
    if (getNumberOfHubs == 1)
      updateLocalModel(mDesc)
    else {
      parameterTree.put((mDesc.getBucket.getStart.toInt, mDesc.getBucket.getEnd.toInt), mDesc.getParams)
      if (parameterTree.size == getNumberOfHubs) {
        mDesc.setParams(
          DenseVector(
            parameterTree.values
              .map(
                {
                  case dense: DenseVector => dense
                  case sparse: SparseVector => sparse.toDenseVector
                })
              .fold(Array[Double]())(
                (accum, vector) => accum.asInstanceOf[Array[Double]] ++ vector.asInstanceOf[DenseVector].data)
              .asInstanceOf[Array[Double]]
          )
        )
        updateLocalModel(mDesc)
      }
    }
  }

}