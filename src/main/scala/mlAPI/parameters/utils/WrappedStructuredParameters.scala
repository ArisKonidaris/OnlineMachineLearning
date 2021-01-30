package mlAPI.parameters.utils

import ControlAPI.CountableSerial

case class WrappedStructuredParameters(var structure: CountableSerial) extends SerializableParameters {

  def this() = this(null)

  def getStructure: CountableSerial = structure

  def setStructure(structure: CountableSerial): Unit = this.structure = structure

  override def getSize: Int = {
    if (structure != null) structure.getSize else 0
  }

}
