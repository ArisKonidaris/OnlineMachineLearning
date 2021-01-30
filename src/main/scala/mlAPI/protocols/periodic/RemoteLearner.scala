package mlAPI.protocols.periodic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait RemoteLearner {

  @RemoteOp
  def updateModel(model: ParameterDescriptor): Unit

}
