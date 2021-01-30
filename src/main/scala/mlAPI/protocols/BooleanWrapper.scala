package mlAPI.protocols

import ControlAPI.CountableSerial

case class BooleanWrapper(var boolean: Boolean) extends CountableSerial {

  def getBoolean: Boolean = boolean

  def setBoolean(boolean: Boolean): Unit = this.boolean = boolean

  override def getSize: Int = 4
}
