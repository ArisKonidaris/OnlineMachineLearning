package mlAPI.parameters.utils

import mlAPI.math.Vector

case class WrappedVectoredParameters(var sizes: Array[Int], var bucket: Bucket, var data: Vector)
  extends SerializableParameters {

  def this() = this(null, null, null)

  def getSizes: Array[Int] = sizes

  def getBucket: Bucket = bucket

  def getData: Vector = data

  def setSizes(sizes: Array[Int]): Unit = this.sizes = sizes

  def setBucket(bucket: Bucket): Unit = this.bucket = bucket

  def setData(data: Vector): Unit = this.data = data

  override def getSize: Int = {
    {
      if (sizes != null) 4 * sizes.length else 0
    } + {
      if (data != null) data.getSize else 0
    } + {
      if (bucket != null) bucket.getSize else 0
    }
  }

}
