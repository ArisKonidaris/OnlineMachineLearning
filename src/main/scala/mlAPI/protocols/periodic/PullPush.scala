package mlAPI.protocols.periodic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait PullPush {

  @RemoteOp
  def pull(): Unit

  @RemoteOp
  def push(modelDescriptor: ParameterDescriptor): Unit

}
