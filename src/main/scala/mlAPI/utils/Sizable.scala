package mlAPI.utils

/** A basic trait for sizable objects. */
trait Sizable extends java.io.Serializable {

  /** Should return the size of the object that extends this trait in Bytes. */
  def getSize: Int

}
