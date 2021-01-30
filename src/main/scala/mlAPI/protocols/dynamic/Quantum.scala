package mlAPI.protocols.dynamic

import ControlAPI.CountableSerial

case class Quantum(var quantum: Double) extends ValueContainer[Double] with CountableSerial {

  require(quantum >= 0.0)

  def this() = this(0.0)

  override def getValue: Double = quantum

  override def setValue(value: Double): Unit = {
    if (quantum >= 0.0) this.quantum = quantum
  }

  override def getSize: Int = 8

}
