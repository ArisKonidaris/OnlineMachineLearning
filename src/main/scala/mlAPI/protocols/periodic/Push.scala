package mlAPI.protocols.periodic

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.utils.ParameterDescriptor

@RemoteProxy
trait Push extends Serializable {

  @RemoteOp
  def push(modelDescriptor: ParameterDescriptor): Unit

}
