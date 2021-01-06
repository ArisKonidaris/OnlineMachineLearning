package mlAPI.protocols.periodic.synchronous

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper
import mlAPI.protocols.ProtocolStatistics

case class SynchronousStatistics() extends ProtocolStatistics {

  /** The name of the distributed training protocol. */
  override protected var protocol: String = "SynchronousAveraging"

  /** The number of models shipped through the network during the distributed training procedure. */
  override protected var modelsShipped: Long = 0

  /** The total number of bytes sent over the network during the distributed training procedure. */
  override protected var bytesShipped: Long = 0

  /** The total number of times that any worker within the network has stop training while waiting for a new model. */
  override protected var numOfBlocks: Long = 0

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable SynchronousStatistics instance."
    }
  }

  @throws[JsonProcessingException]
  def toJsonString: String = new ObjectMapper().writerWithDefaultPrettyPrinter.writeValueAsString(this)

}
