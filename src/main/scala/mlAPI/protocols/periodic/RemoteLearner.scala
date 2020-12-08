package mlAPI.protocols.periodic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait RemoteLearner {

  @RemoteOp
  def updateModel(model: ParameterDescriptor): Unit

}
