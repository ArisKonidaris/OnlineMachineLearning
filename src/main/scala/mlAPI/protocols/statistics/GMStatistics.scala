package mlAPI.protocols.statistics

import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

case class GMStatistics() extends ProtocolStatistics {

  /** The name of the distributed training protocol. */
  override protected var protocol: String = "GM"

  /** The number of rounds of the MG distributed learning protocol. */
  protected var numOfRounds: Long = 0

  /** The number of sub-rounds of the MG distributed learning protocol. */
  protected var numOfSubRounds: Long = 0

  def getNumOfRounds: Long = numOfRounds

  def getNumOfSubRounds: Long = numOfSubRounds

  def setNumOfRounds(numOfRounds: Long): GMStatistics = {
    this.numOfRounds = numOfRounds
    this
  }

  def setNumOfSubRounds(numOfSubRounds: Long): GMStatistics = {
    this.numOfSubRounds = numOfSubRounds
    this
  }

  def updateNumOfRounds(update: Long = 1): Unit = numOfRounds = incrementLong(numOfRounds, update)

  def updateNumOfSubRounds(update: Long = 1): Unit = numOfSubRounds = incrementLong(numOfSubRounds, update)

  @JsonIgnore
  override def clear(): Unit = {
    super.clear()
    numOfRounds = 0
    numOfSubRounds = 0
  }

  @JsonIgnore
  override def getSize: Int = super.getSize + 16

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable GMStatistics instance."
    }
  }

  @throws[JsonProcessingException]
  def toJsonString: String = new ObjectMapper().writerWithDefaultPrettyPrinter.writeValueAsString(this)

}
