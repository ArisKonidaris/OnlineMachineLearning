package mlAPI.protocols.periodic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait Push extends Serializable {

  @RemoteOp
  def pushModel(modelDescriptor: ParameterDescriptor): Unit

}
