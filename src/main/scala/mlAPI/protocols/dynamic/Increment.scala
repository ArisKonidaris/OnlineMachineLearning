package mlAPI.protocols.dynamic

import ControlAPI.CountableSerial

case class Increment(var increment: Long, var subRound: Long) extends java.io.Serializable with CountableSerial {

  def setIncrement(increment: Long): Unit = this.increment = increment

  def setSubRound(subRound: Long): Unit = this.subRound = subRound

  def getIncrement: Long = increment

  def getSubRound: Long = subRound

  override def getSize: Int = 16

}
