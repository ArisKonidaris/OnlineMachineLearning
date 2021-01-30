package mlAPI.protocols.dynamic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait GMHubInterface {

  @RemoteOp
  def pull(): Unit

  @RemoteOp
  def endWarmup(modelDescriptor: ParameterDescriptor): Unit

  @RemoteOp
  def violation(): Unit

  @RemoteOp
  def receiveLocalModel(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor]

}
