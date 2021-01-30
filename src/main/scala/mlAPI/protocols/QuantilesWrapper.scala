package mlAPI.protocols

import ControlAPI.CountableSerial
import mlAPI.parameters.utils.Bucket

case class QuantilesWrapper(var quantiles: Array[Array[Bucket]]) extends CountableSerial {

  def this() = this(null)

  def getQuantiles: Array[Array[Bucket]] = quantiles

  def setQuantiles(quantiles: Array[Array[Bucket]]): Unit = this.quantiles = quantiles

  override def getSize: Int = {
    if (quantiles != null)
      (for (x <- quantiles) yield (for (y <- x) yield y.getSize).sum).sum
    else
      0
  }
}
