package mlAPI.protocols

import ControlAPI.CountableSerial

case class DoubleWrapper(var double: Double) extends CountableSerial {

  def this() = this(0D)

  def getDouble: Double = double

  def setDouble(double: Double): Unit = this.double = double

  override def getSize: Int = 8
}
