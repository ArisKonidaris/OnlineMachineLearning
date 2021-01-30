package mlAPI.protocols

import ControlAPI.CountableSerial

case class IntWrapper(var int: Int) extends CountableSerial {

  def this() = this(0)

  def getInt: Int = int

  def setInt(int: Int): Unit = this.int = int

  override def getSize: Int = 4
}
