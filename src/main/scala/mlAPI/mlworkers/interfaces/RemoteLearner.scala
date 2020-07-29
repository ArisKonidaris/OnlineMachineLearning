package mlAPI.mlworkers.interfaces

import BipartiteTopologyAPI.annotations.{DefaultOp, RemoteOp, RemoteProxy}
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait RemoteLearner {

  @RemoteOp
  def updateModel(model: ParameterDescriptor): Unit

}
