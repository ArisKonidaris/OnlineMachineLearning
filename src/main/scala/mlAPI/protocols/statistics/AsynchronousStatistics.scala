package mlAPI.protocols.statistics

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

case class AsynchronousStatistics() extends ProtocolStatistics {

  /** The name of the distributed training protocol. */
  override protected var protocol: String = "AsynchronousAveraging"

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable AsynchronousStatistics instance."
    }
  }

  @throws[JsonProcessingException]
  def toJsonString: String = new ObjectMapper().writerWithDefaultPrettyPrinter.writeValueAsString(this)

}
