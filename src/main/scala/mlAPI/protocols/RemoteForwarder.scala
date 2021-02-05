package mlAPI.protocols

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}

@RemoteProxy
trait RemoteForwarder {

  @RemoteOp
  def poll(): Unit

}
