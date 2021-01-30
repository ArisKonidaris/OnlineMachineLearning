package mlAPI.protocols.dynamic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait GMRemoteLearner {

  @RemoteOp
  def updateModel(model: ParameterDescriptor): Unit

  @RemoteOp
  def sendLocalModel(): Unit

}
