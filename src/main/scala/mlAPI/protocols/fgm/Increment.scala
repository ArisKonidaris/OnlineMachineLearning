package mlAPI.protocols.fgm

import mlAPI.utils.Sizable

case class Increment(var increment: Long, var subRound: Long) extends java.io.Serializable with Sizable {

  def setIncrement(increment: Long): Unit = this.increment = increment

  def setSubRound(subRound: Long): Unit = this.subRound = subRound

  def getIncrement: Long = increment

  def getSubRound: Long = subRound

  /** Should return the size of the object that extends this trait in Bytes. */
  override def getSize: Int = 16

}
