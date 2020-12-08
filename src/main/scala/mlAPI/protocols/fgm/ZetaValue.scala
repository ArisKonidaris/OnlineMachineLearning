package mlAPI.protocols.fgm

import mlAPI.utils.Sizable

case class ZetaValue(var zeta: Double) extends ValueContainer[Double] with Sizable {

  def this() = this(0.0)

  override def getValue: Double = zeta

  override def setValue(value: Double): Unit = this.zeta = zeta

  /** Should return the size of the object that extends this trait in Bytes. */
  override def getSize: Int = 8
}
