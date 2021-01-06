package mlAPI.protocols.fgm

import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper
import mlAPI.protocols.ProtocolStatistics

case class FGMStatistics() extends ProtocolStatistics {

  /** The name of the distributed training protocol. */
  override protected var protocol: String = "FGM"

  /** The number of models shipped through the network during the distributed training procedure. */
  override protected var modelsShipped: Long = 0

  /** The total number of bytes sent over the network during the distributed training procedure. */
  override protected var bytesShipped: Long = 0

  /** The total number of times that any worker within the network has stop training while waiting for a new model. */
  override protected var numOfBlocks: Long = 0

  /** The number of rounds of the FMG distributed learning protocol. */
  protected var numOfRounds: Long = 0

  /** The number of sub-rounds of the FMG distributed learning protocol. */
  protected var numOfSubRounds: Long = 0

  def getNumOfRounds: Long = numOfRounds

  def getNumOfSubRounds: Long = numOfSubRounds

  def setNumOfRounds(numOfRounds: Long): FGMStatistics = {
    this.numOfRounds = numOfRounds
    this
  }

  def setNumOfSubRounds(numOfSubRounds: Long): FGMStatistics = {
    this.numOfSubRounds = numOfSubRounds
    this
  }

  def updateNumOfRounds(update: Long = 1): Unit = numOfRounds = incrementLong(numOfRounds, update)

  def updateNumOfSubRounds(update: Long = 1): Unit = numOfSubRounds = incrementLong(numOfSubRounds, update)

  @JsonIgnore
  override def getSize: Int = super.getSize + 16

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
