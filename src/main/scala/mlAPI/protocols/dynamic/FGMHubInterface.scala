package mlAPI.protocols.dynamic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait FGMHubInterface {

  @RemoteOp
  def pull(): Unit

  @RemoteOp
  def endWarmup(modelDescriptor: ParameterDescriptor): Unit

  @RemoteOp
  def receiveIncrement(increment: Increment): Unit

  @RemoteOp
  def receiveZeta(zeta: ZetaValue): Unit

  @RemoteOp
  def receiveLocalDrift(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor]

}
