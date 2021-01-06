package mlAPI.parameters

import mlAPI.math.{DenseVector, Vector}

class SerializedVectoredParameters(var sizes: Array[Int], var data: Vector) extends SerializedParameters {

  def this() = this(Array[Int](), DenseVector())

  def getSizes: Array[Int] = sizes

  def getData: Vector = data

  def setSizes(sizes: Array[Int]): Unit = this.sizes = sizes

  def setData(data: Vector): Unit = this.data = data

  override def getSize: Int = { if (sizes != null) 4 * sizes.length else 0 } + { if (data != null) data.getSize else 0 }

}
