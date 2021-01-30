package mlAPI.protocols

import ControlAPI.CountableSerial

case class LongWrapper(var long: Long) extends CountableSerial {

  def this() = this(0)

  def getLong: Long = long

  def setLong(long: Long): Unit = this.long = long

  override def getSize: Int = 8

}
