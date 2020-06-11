package mlAPI.mlworkers.interfaces

import ControlAPI.QueryResponse
import BipartiteTopologyAPI.annotations.RemoteOp

trait Querier {

  @RemoteOp
  def sendQueryResponse(qResponse: QueryResponse): Unit

}
