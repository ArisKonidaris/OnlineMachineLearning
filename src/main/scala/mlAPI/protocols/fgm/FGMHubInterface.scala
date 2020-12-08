package mlAPI.protocols.fgm

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait FGMHubInterface {

  @RemoteOp
  def pullModel: Response[ParameterDescriptor]

  @RemoteOp
  def endWarmup(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor]

  @RemoteOp
  def receiveIncrement(increment: Increment): Unit

  @RemoteOp
  def receiveZeta(zeta: ZetaValue): Unit

  @RemoteOp
  def receiveLocalDrift(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor]

}
