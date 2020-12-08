package mlAPI.protocols.fgm

/** A Serializable trait of a value. */
trait ValueContainer[T] extends java.io.Serializable {

  def getValue: T

  def setValue(value: T): Unit

}
