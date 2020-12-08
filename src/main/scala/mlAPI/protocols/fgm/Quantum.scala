package mlAPI.protocols.fgm

import mlAPI.utils.Sizable

case class Quantum(var quantum: Double) extends ValueContainer[Double] with Sizable {

  require(quantum >= 0.0)

  def this() = this(0.0)

  override def getValue: Double = quantum

  override def setValue(value: Double): Unit = {
    if (quantum >= 0.0) this.quantum = quantum
  }

  /** Should return the size of the object that extends this trait in Bytes. */
  override def getSize: Int = 8
}
