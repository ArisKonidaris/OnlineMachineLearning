package mlAPI.protocols.periodic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait PushPull extends Serializable {

  @RemoteOp
  def pull(): Unit

  @RemoteOp
  def pushPull(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor]

  @RemoteOp
  def push(modelDescriptor: ParameterDescriptor): Unit

}
