package mlAPI.protocols

import ControlAPI.CountableSerial
import com.fasterxml.jackson.annotation.JsonIgnore

/**
 * A basic trait for holding the basic statistics of a distributed training procedure.
 */
trait ProtocolStatistics extends CountableSerial {

  /** The name of the distributed training protocol. */
  protected var protocol: String

  /** The number of models shipped through the network during the distributed training procedure. */
  protected var modelsShipped: Long

  /** The total number of bytes sent over the network during the distributed training procedure. */
  protected var bytesShipped: Long

  /** The total number of times that any worker within the network has stop training while waiting for a new model. */
  protected var numOfBlocks: Long

  def getProtocol: String = {
    val value = protocol
    value
  }

  def getModelsShipped: Long = {
    val value = modelsShipped
    value
  }

  def getBytesShipped: Long = {
    val value = bytesShipped
    value
  }

  def getNumOfBlocks: Long = {
    val value = numOfBlocks
    value
  }

  def setProtocol(protocol: String): ProtocolStatistics = {
    this.protocol = protocol
    this
  }

  def setModelsShipped(modelsShipped: Long): ProtocolStatistics = {
    this.modelsShipped = modelsShipped
    this
  }

  def setBytesShipped(bytesShipped: Long): ProtocolStatistics = {
    this.bytesShipped = bytesShipped
    this
  }

  def setNumOfBlocks(numOfBlocks: Long): ProtocolStatistics = {
    this.numOfBlocks = numOfBlocks
    this
  }

  def updateModelsShipped(update: Long = 1): Unit = modelsShipped = incrementLong(modelsShipped, update)

  def updateBytesShipped(update: Long = 1): Unit = bytesShipped = incrementLong(bytesShipped, update)

  def updateNumOfBlocks(update: Long = 1): Unit = numOfBlocks = incrementLong(numOfBlocks, update)

  def incrementLong(variable: Long, size: Long): Long = {
    if (variable != Long.MaxValue)
      if (variable < Long.MaxValue - size)
        variable + size
      else
        Long.MaxValue
    else
      variable
  }

  @JsonIgnore
  override def getSize: Int = {
    if (protocol != null) 8 * protocol.length else 0
  } + 24

}
