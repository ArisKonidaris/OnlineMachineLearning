package mlAPI.learners.classification.trees.serializable

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper
import mlAPI.learners.classification.trees.serializable.nodes.NodeDescriptor

/**
 * A serializable descriptor of a Hoeffding Tree.
 */
case class HTDescriptor(var discrete: Boolean,
                        var maxByteSize: Int,
                        var n_min: Long,
                        var tau: Double,
                        var delta: Double,
                        var num_of_classes: Double,
                        var splits: Int,
                        var mem_period: Int,
                        var classes: Array[Double],
                        var targets: Array[Int],
                        var range: Double,
                        var maxLabel: Int,
                        var size: Int,
                        var tree: NodeDescriptor,
                        var height: Int,
                        var numberOfInternalNodes: Int,
                        var numberOfLeaves: Int,
                        var inactiveLeaves: Int,
                        var activeLeaves: Int,
                        var dataPointsSeen: Long,
                        var leafCounter: Int,
                        var activeSize: Int,
                        var inactiveSize: Int)
  extends java.io.Serializable {

  def setDiscrete(discrete: Boolean): Unit = this.discrete = discrete

  def getDiscrete: Boolean = discrete

  def setMaxByteSize(maxByteSize: Int): Unit = this.maxByteSize = maxByteSize

  def getMaxByteSize: Int = maxByteSize

  def setNMin(n_min: Long): Unit = this.n_min = n_min

  def getNMin: Long = n_min

  def setTau(tau: Double): Unit = this.tau = tau

  def getTau: Double = tau

  def setDelta(delta: Double): Unit = this.delta = delta

  def getDelta: Double = delta

  def setNumOfClasses(num_of_classes: Double): Unit = this.num_of_classes = num_of_classes

  def getNumOfClasses: Double = num_of_classes

  def setSplits(splits: Int): Unit = this.splits = splits

  def getSplits: Int = splits

  def setMemPeriod(mem_period: Int): Unit = this.mem_period = mem_period

  def getMemPeriod: Int = mem_period

  def setClasses(classes: Array[Double]): Unit = this.classes = classes

  def getClasses: Array[Double] = classes

  def setTargets(targets: Array[Int]): Unit = this.targets = targets

  def getTargets: Array[Int] = targets

  def setRange(range: Double): Unit = this.range = range

  def getRange: Double = range

  def setMaxLabel(maxLabel: Int): Unit = this.maxLabel = maxLabel

  def getMaxLabel: Int = maxLabel

  def setSize(size: Int): Unit = this.size = size

  def getSize: Int = size

  def setTree(tree: NodeDescriptor): Unit = this.tree = tree

  def getTree: NodeDescriptor = tree

  def setHeight(height: Int): Unit = this.height = height

  def getHeight: Int = height

  def setNumberOfInternalNodes(numberOfInternalNodes: Int): Unit = this.numberOfInternalNodes = numberOfInternalNodes

  def getNumberOfInternalNodes: Int = numberOfInternalNodes

  def setNumberOfLeaves(numberOfLeaves: Int): Unit = this.numberOfLeaves = numberOfLeaves

  def getNumberOfLeaves: Int = numberOfLeaves

  def setInactiveLeaves(inactiveLeaves: Int): Unit = this.inactiveLeaves = inactiveLeaves

  def getInactiveLeaves: Int = inactiveLeaves

  def setActiveLeaves(activeLeaves: Int): Unit = this.activeLeaves = activeLeaves

  def getActiveLeaves: Int = activeLeaves

  def setDataPointsSeen(dataPointsSeen: Long): Unit = this.dataPointsSeen = dataPointsSeen

  def getDataPointsSeen: Long = dataPointsSeen

  def setLeafCounter(leafCounter: Int): Unit = this.leafCounter = leafCounter

  def getLeafCounter: Int = leafCounter

  def setActiveSize(activeSize: Int): Unit = this.activeSize = activeSize

  def getActiveSize: Int = activeSize

  def setInactiveSize(inactiveSize: Int): Unit = this.inactiveSize = inactiveSize

  def getInactiveSize: Int = inactiveSize

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable " + this.getClass.getName
    }
  }

  def toJsonString: String = {
    new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(this)
  }

}
