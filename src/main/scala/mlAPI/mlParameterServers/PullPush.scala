package mlAPI.mlParameterServers

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import BipartiteTopologyAPI.futures.Response
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait PullPush extends Serializable {

  @RemoteOp
  def pullModel: Response[ParameterDescriptor]

  @RemoteOp
  def pushModel(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor]

}
