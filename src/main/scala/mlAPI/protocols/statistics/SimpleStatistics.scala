package mlAPI.protocols.statistics

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

case class SimpleStatistics() extends ProtocolStatistics {

  /** The name of the distributed training protocol. */
  override protected var protocol: String = "Centralized"

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable SimpleStatistics instance."
    }
  }

  @throws[JsonProcessingException]
  def toJsonString: String = new ObjectMapper().writerWithDefaultPrettyPrinter.writeValueAsString(this)

}
