package mlAPI.mlworkers.interfaces

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait MLWorkerRemote {

  @RemoteOp
  def updateModel(model: ParameterDescriptor): Unit

}
