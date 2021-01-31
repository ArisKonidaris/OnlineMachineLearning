package mlAPI.mlworkers.interfaces

import ControlAPI.CountableSerial
import BipartiteTopologyAPI.annotations.RemoteOp

trait Querier {

  @RemoteOp
  def sendQueryResponse(qResponse: CountableSerial): Unit

}
