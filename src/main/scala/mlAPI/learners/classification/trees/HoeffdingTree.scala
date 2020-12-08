package mlAPI.learners.classification.trees

import com.google.common.collect.{BiMap, HashBiMap}
import mlAPI.math.{Vector, LabeledPoint, UnlabeledPoint}
import nodes.{InternalNode, LeafNode, Node, TestAttribute}
import serializable.HTDescriptor
import serializable.nodes.{InternalNodeDescriptor, LeafNodeDescriptor}
import stats.{DiscreteStatistics, NumericalStatistics}

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * A Hoeffding tree classifier.
 *
 * @param discrete       A boolean determining if the Tree will be trained on discrete or numerical features.
 * @param maxByteSize    Maximum memory consumed by the tree.
 * @param n_min          Grace period.
 * @param tau            Tie breaking parameter.
 * @param delta          Probability of error of the Hoeffding bound.
 * @param num_of_classes The total number of classes for the classification problem.
 * @param splits         The number of splits to test upon each attribute.
 * @param mem_period     The number of data points trained between each memory scan of the Hoeffding tree.
 */
class HoeffdingTree(private var discrete: Boolean = false,
                    private var maxByteSize: Int = 33554432,
                    private var n_min: Long = 200,
                    private var tau: Double = 0.05,
                    private var delta: Double = 1.0E-7D,
                    private var num_of_classes: Double = 2,
                    private var splits: Int = 10,
                    private var mem_period: Int = 100000) {

  private val classMap: BiMap[Double, Int] = HashBiMap.create[Double, Int]()
  private val leafMap: mutable.Map[String, LeafNode] = mutable.HashMap[String, LeafNode]()
  private var range: Double = Math.log(num_of_classes) / Math.log(2)
  private var maxLabel: Int = -1
  private var size: Int = 0
  private var tree: Node = _
  private var height: Int = 0
  private var numberOfInternalNodes: Int = 0
  private var numberOfLeaves: Int = 0
  private var inactiveLeaves: Int = 0
  private var activeLeaves: Int = 0
  private var dataPointsSeen: Long = 0
  private var leafCounter: Int = 0
  private var activeSize: Int = 0
  private var inactiveSize: Int = 0


  ///////////////////////////////////////////// Parameter Validation ///////////////////////////////////////////////////

  require(maxByteSize > 0 && maxByteSize <= 2147483647)
  require(n_min >= 0 && n_min <= 2147483647)
  require(tau >= 0.0D && tau <= 1.0D)
  require(delta >= 0.0D && delta <= 1.0D)
  require(num_of_classes > 0 && num_of_classes <= 2147483647)
  require(splits > 0 && splits < 1000)
  require(mem_period >= 0 && mem_period <= 2147483647)


  ////////////////////////////////////////////////// Setters ///////////////////////////////////////////////////////////

  def setDiscrete(discrete: Boolean): HoeffdingTree = {
    this.discrete = discrete
    this
  }

  def setMaxByteSize(maxByteSize: Int): HoeffdingTree = {
    this.maxByteSize = maxByteSize
    this
  }

  def setNMin(n_min: Long): HoeffdingTree = {
    this.n_min = n_min
    this
  }

  def setTau(tau: Double): HoeffdingTree = {
    this.tau = tau
    this
  }

  def setDelta(delta: Double): HoeffdingTree = {
    this.delta = delta
    this
  }

  def setHeight(height: Int): Unit = this.height = height

  def setNumberOfInternalNodes(numberOfInternalNodes: Int): Unit = this.numberOfInternalNodes = numberOfInternalNodes

  def setNumberOfLeaves(numberOfLeaves: Int): Unit = this.numberOfLeaves = numberOfLeaves

  def setActiveLeaves(activeLeaves: Int): Unit = this.activeLeaves = activeLeaves

  def setNumOfClasses(num_of_classes: Double): HoeffdingTree = {
    this.num_of_classes = num_of_classes
    range = Math.log(num_of_classes) / Math.log(2)
    this
  }

  def setSplits(splits: Int): HoeffdingTree = {
    this.splits = splits
    this
  }

  def setMemPeriod(mem_period: Int): HoeffdingTree = {
    this.mem_period = mem_period
    this
  }

  def setNumberOfDataPointsSeen(dataPointsSeen: Long): HoeffdingTree = {
    this.dataPointsSeen = dataPointsSeen
    this
  }

  def setLeafCounter(leafCounter: Int): HoeffdingTree = {
    this.leafCounter = leafCounter
    this
  }


  ////////////////////////////////////////////////// Getters ///////////////////////////////////////////////////////////

  def getLeafMap: mutable.Map[String, LeafNode] = leafMap

  def getMaxByteSize: Int = {
    val value: Int = maxByteSize
    value
  }

  def getRange: Double = {
    val value: Double = range
    value
  }

  def getDelta: Double = {
    val value: Double = delta
    value
  }

  def getTau: Double = {
    val value: Double = tau
    value
  }

  def getHeight: Int = {
    val value: Int = height
    value
  }

  def getNumberOfLeaves: Int = {
    val value: Int = numberOfLeaves
    value
  }

  def getNumberOfInternalNodes: Int = {
    val value: Int = numberOfInternalNodes
    value
  }

  def getNumberOfActiveLeaves: Int = {
    val value: Int = activeLeaves
    value
  }

  def getSize: Int = {
    val value: Int = size
    value
  }

  def getNumberOfDataPointsSeen: Long = {
    val value: Long = dataPointsSeen
    value
  }

  def getLeafCounter: Int = {
    val value: Int = leafCounter
    value
  }

  def getMemPeriod: Int = {
    val value:Int = mem_period
    value
  }


  ///////////////////////////////////////////// Auxiliary methods //////////////////////////////////////////////////////

  /**
   * Overflow safe incrementing counter of number of data points processed.
   */
  def incrementNumberOfDataProcessed(): Unit = {
    if (dataPointsSeen + 1 == Long.MaxValue)
      dataPointsSeen = 0
    else
      dataPointsSeen += 1
  }

  /**
   * Overflow safe incrementing counter of leaves. This counter is used for maintaining a unique id upon the leaves.
   */
  def incrementLeafCounter(): Unit = if (leafCounter == Int.MaxValue) leafCounter = 0 else leafCounter += 1

  /**
   * Checks if the Hoeffding Tree is too large to train.
   *
   * @return A flag determining if the Hoeffding Tree can continue training.
   */
  def canTrain: Boolean = {
    if (inactiveLeaves == numberOfLeaves && numberOfLeaves > 0) {
      println("The Hoeffding tree cannot fit any more data points. Memory limit reached. All leaves are inactive.")
      return false
    }
    true
  }

  /**
   * A method for returning a visual representation of the Hoeffding Tree.
   *
   * @return A String representation of the Hoeffding Tree.
   */
  override def toString: String = {
    "\n\nTrained on " + dataPointsSeen + " data points" + "\n" +
      "Height : " + height + "\n" +
      "Number of Internal Nodes: " + numberOfInternalNodes + "\n" +
      "Number of Leaves: " + numberOfLeaves + "\n" +
      "Number of active leaves: " + activeLeaves + "\n" +
      "Number of inactive leaves: " + inactiveLeaves + "\n\n" + {
      if (tree != null) tree.toString else "{ }"
    } + "\n\n"
  }

  /**
   * Serialize the Hoeffding Tree.
   *
   * @return A Serializable Hoeffding Tree.
   */
  def serialize: HTDescriptor = {

    val it = classMap.entrySet().iterator()
    val cl: ListBuffer[Double] = ListBuffer[Double]()
    val tar: ListBuffer[Int] = ListBuffer[Int]()
    while (it.hasNext) {
      val item = it.next()
      cl += item.getKey
      tar += item.getValue
    }

    HTDescriptor(
      discrete,
      maxByteSize,
      n_min,
      tau,
      delta,
      num_of_classes,
      splits,
      mem_period,
      cl.toArray,
      tar.toArray,
      range,
      maxLabel,
      size,
      tree.serialize,
      height,
      numberOfInternalNodes,
      numberOfLeaves,
      inactiveLeaves,
      activeLeaves,
      dataPointsSeen,
      leafCounter,
      activeSize,
      inactiveSize
    )
  }

  /**
   * Deserialize a Hoeffding Tree Descriptor instance into a Hoeffding Tree instance.
   *
   * @param descriptor A Serializable Hoeffding Tree.
   */
  def deserialize(descriptor: HTDescriptor): Unit = {
    tree = {
      descriptor.tree match {
        case ind: InternalNodeDescriptor => InternalNode.deserialize(ind)
        case lnd: LeafNodeDescriptor => LeafNode.deserialize(lnd)
        case _ => throw new RuntimeException("Unknown serialization scheme.")
      }
    }
    discrete = descriptor.getDiscrete
    maxByteSize = descriptor.getMaxByteSize
    n_min = descriptor.getNMin
    tau = descriptor.getTau
    delta = descriptor.getDelta
    num_of_classes = descriptor.getNumOfClasses
    splits = descriptor.getSplits
    mem_period = descriptor.getMemPeriod
    classMap.clear()
    for ((cl, tar) <- descriptor.getClasses.zip(descriptor.getTargets)) classMap.put(cl, tar)
    leafMap.clear()
    tree.createLeafMap(leafMap)
    range = descriptor.getRange
    maxLabel = descriptor.getMaxLabel
    size = descriptor.getSize
    height = descriptor.getHeight
    numberOfInternalNodes = descriptor.getNumberOfInternalNodes
    numberOfLeaves = descriptor.getNumberOfLeaves
    inactiveLeaves = descriptor.getInactiveLeaves
    activeLeaves = descriptor.getActiveLeaves
    dataPointsSeen = descriptor.getDataPointsSeen
    leafCounter = descriptor.getLeafCounter
    activeSize = descriptor.getActiveSize
    inactiveSize = descriptor.getInactiveSize
  }

  ////////////////////////////////////////////////// Methods ///////////////////////////////////////////////////////////

  /**
   * Clears the Hoeffding Tree classifier.
   */
  def clear(): Unit = {
    classMap.clear()
    leafMap.clear()
    maxLabel = -1
    size = 0
    tree = null
    height = 0
    numberOfInternalNodes = 0
    numberOfLeaves = 0
    activeLeaves = 0
    inactiveLeaves = 0
    dataPointsSeen = 0
    leafCounter = 0
    activeSize = 0
    inactiveSize = 0
  }

  /**
   * Calculates the size of the Hoeffding Tree.
   */
  def calculateTreeSize: Int = (for (leaf <- leafMap) yield leaf._2.getNodeSize).toArray.sum + numberOfInternalNodes * 16

  /**
   * Calculates the size of the Hoeffding Tree along with its auxiliary parameters.
   */
  def calculateSize(): Unit = size = 101 + classMap.size * 12 + calculateTreeSize

  /**
   * A method for updating the leaf HashMap after each leaf split.
   *
   * @param oldLeaf The old leaf that is split.
   * @param newNode The new generated internal node with its two child nodes.
   */
  def updateLeafMap(oldLeaf: LeafNode, newNode: InternalNode): Unit = {
    leafMap.remove(oldLeaf.id + "_" + oldLeaf.height)
    val left_child: LeafNode = newNode.leftChild.asInstanceOf[LeafNode]
    val right_child: LeafNode = newNode.rightChild.asInstanceOf[LeafNode]
    leafMap.put(left_child.id + "_" + left_child.height, left_child)
    leafMap.put(right_child.id + "_" + right_child.height, right_child)
  }

  /**
   * This method prepared a [[LabeledPoint]] instance for training the Hoeffding Tree.
   *
   * @param point The labeled data point used to train the Hoeffding tree.
   * @return A two tuple containing the feature vector and the class.
   */
  def preprocessPoint(point: LabeledPoint): (Vector, Int) = {
    if (!classMap.containsKey(point.label)) {
      maxLabel += 1
      classMap.put(point.label, maxLabel)
    }
    (if (discrete) point.discreteVector else point.numericVector, classMap.get(point.label))
  }

  /**
   * Filter the data point through the Hoeffding Tree.
   *
   * @param vector The features vector to filter in the Hoeffding Tree.
   * @return The leaf node where the feature vector reaches within the Hoeffding Tree.
   */
  def filterLeaf(vector: Vector): LeafNode = {
    try {
      tree.filterNode(vector)
    } catch {
      case _: Exception =>
        val leafNode: LeafNode = new LeafNode(
          1,
          false,
          if (discrete) DiscreteStatistics() else NumericalStatistics(splits),
          null,
          1
        )
        tree = leafNode
        leafMap.put(1 + "_" + 1, leafNode)
        numberOfLeaves += 1
        activeLeaves += 1
        height = 1
        leafCounter += 1
        leafNode
    }
  }

  /**
   * A method for updating the sufficient statistics of a [[LeafNode]].
   *
   * @param leaf        An leaf of the Hoeffding Tree. This should be the leaf that a training data point is filtered into.
   * @param dataPoint   A data point to train the leaf on.
   * @param target      The label of the data point.
   * @param splitting   A flag that enables the leaf to split if necessary.
   * @param memoryCheck A flag that enables the memory management of the Hoeffding Tree.
   */
  def updateLeaf(leaf: LeafNode,
                 dataPoint: Vector,
                 target: Int,
                 splitting: Boolean = true,
                 memoryCheck: Boolean = true): Int = {
    var error: Int = 0
    if (leaf.predict(dataPoint)._1 != target) {
      // Update the error counter of the filtered leaf.
      leaf.errors += 1
      error += 1
    }
    leaf.stats.updateStats(dataPoint, target) // Update the sufficient statistics of the filtered leaf.
    if (leaf.stats.isActive) {
      leaf.n += 1 // Update the data point counter of the filtered leaf.
      if (splitting) if (leaf.n % n_min == 0) split(leaf, memoryCheck) // Check for split.
    }
    error
  }

  /**
   * This method computes the Hoeffding Bound and splits, if necessary, the current leaf node.
   *
   * @param leaf        The leaf to try to split.
   * @param checkMemory A flag that enables the memory management of the Hoeffding Tree.
   */
  def split(leaf: LeafNode, checkMemory: Boolean = true): Unit = {

    // Calculating the Hoeffding bound along with the best splits for the filtered leaf.
    val hoeffdingBound: Double = Math.sqrt((range * range * Math.log(1.0 / delta)) / (2.0 * leaf.stats.n_l))
    val bestSplits: Array[(Int, Double, Double, Array[Array[(Int, Double)]])] = leaf.stats.bestSplits
    val bestSplit = bestSplits.head
    val secondBestSplit = bestSplits.tail.head

    // Check if split is needed.
    if (bestSplit._3 - secondBestSplit._3 > hoeffdingBound || hoeffdingBound < tau) {
      //    if ((bestSplit._3 > hoeffdingBound) && (bestSplit._3 - secondBestSplit._3 > hoeffdingBound || hoeffdingBound < tau)) {

      // Create the new Internal Node along with its two child Leaves.
      val intNode = new InternalNode(new TestAttribute(bestSplit._1, bestSplit._2), leaf.height)
      intNode.leftChild = LeafNode(-1, isLeft = true, leaf.stats.createStats.setStats(bestSplit._4.head), intNode, leaf.height + 1)
      intNode.rightChild = LeafNode(-1, isLeft = false, leaf.stats.createStats.setStats(bestSplit._4.tail.head), intNode, leaf.height + 1)
      if (leaf.parent != null)
        if (leaf.isLeft)
          leaf.parent.leftChild = intNode
        else
          leaf.parent.rightChild = intNode

      // Update the auxiliary data of the tree.
      val h = leaf.height + 1
      if (height < h) height = h
      numberOfInternalNodes += 1
      numberOfLeaves += 1
      activeLeaves += 1
      leafCounter += 2
      intNode.leftChild.asInstanceOf[LeafNode].id = leafCounter - 1
      intNode.rightChild.asInstanceOf[LeafNode].id = leafCounter

      // Update the Leaf List.
      tree match {
        case _: LeafNode =>
          tree = intNode
          updateLeafMap(leaf, intNode)
        case _ => updateLeafMap(leaf, intNode)
      }

      // Check the memory consumption of the Hoeffding Tree.
      if (checkMemory) memCheck()
    }
  }

  /**
   * This method trains the Hoeffding Tree on a labeled data point.
   *
   * @param point The labeled data point for the Hoeffding Tree to be trained on.
   * @param splitting Whether to spit the nodes while training on the labeled data point.
   * @param memoryCheck Whether to perform memory checks or not while training the tree.
   */
  def fit(point: LabeledPoint, splitting: Boolean = true, memoryCheck: Boolean = true): Int = {

    var error: Int = 0

    // Update the counter indicating the total number of data points that the Hoeffding Tree has been trained on.
    incrementNumberOfDataProcessed()

    // Train the Hoeffding Tree if it is not too large.
    if (canTrain) {
      // Prepare the labeled point for learning.
      val (dataPoint, target) = preprocessPoint(point)

      // Filter the data point through the Hoeffding Tree.
      val fNode: LeafNode = filterLeaf(dataPoint)

      // Fit the data point to the filtered leaf and split if necessary.
      error = updateLeaf(fNode, dataPoint, target, splitting, memoryCheck)

      // Update the node size approximations if necessary.
      if (memoryCheck) if (dataPointsSeen % mem_period == 0) updateLeafSizeEstimates()
    }

    error

  }

  /**
   * This method trains the Hoeffding Tree on a batch of labeled data points.
   *
   * @param batch A batch of labeled data points for the Hoeffding Tree to be trained on.
   */
  def fit(batch: ListBuffer[LabeledPoint]): Int = (for (point: LabeledPoint <- batch) yield fit(point)).sum

  /**
   * Updates the memory consumption estimates.
   */
  def updateLeafSizeEstimates(): Unit = {
    val actives = leafMap.toArray.filter(leaf => leaf._2.stats.isActive).map(vk => vk._2)
    val nonactives = leafMap.toArray.filter(leaf => !leaf._2.stats.isActive).map(vk => vk._2)
    assert(activeLeaves == actives.length && inactiveLeaves == nonactives.length)
    activeSize = (for (activeLeaf <- actives) yield activeLeaf.getNodeSize).sum
    activeSize /= activeLeaves
    inactiveSize = (for (activeLeaf <- nonactives) yield activeLeaf.getNodeSize).sum
    inactiveSize = if (inactiveLeaves == 0) 0 else inactiveSize / inactiveLeaves
  }

  /**
   * A method for bounding the size of the Hoeffding Tree (memory management).
   */
  def memCheck(): Unit = {

    @tailrec
    def calculateMaxAvailableActiveNodes(active: Int, inactive: Int): Int = {
      assert(active >= 0 && inactive >= 0 && active + inactive == numberOfLeaves && leafMap.size == numberOfLeaves)
      if (maxByteSize - numberOfInternalNodes * 16 - active * activeSize - inactive * inactiveSize < 0)
        active - 1
      else
        calculateMaxAvailableActiveNodes(active + 1, inactive - 1)
    }

    // Estimating the size of the Hoeffding tree.
    val treeSizeEstimate: Int = activeLeaves * activeSize + inactiveLeaves * inactiveSize + numberOfInternalNodes * 16
    if (inactiveLeaves > 0 || treeSizeEstimate > maxByteSize) {

      // Free space for active leaves.
      var supportedActiveNodes = calculateMaxAvailableActiveNodes(0, numberOfLeaves)

      // Dynamically activating and deactivating leaves.
      val sortedLeaves: Array[LeafNode] = leafMap.toArray.map(vk => vk._2).sortBy(x => -x.errors)
      for (leaf <- sortedLeaves) {
        if (supportedActiveNodes > 0) {
          if (!leaf.stats.isActive) {
            leaf.activate()
            activeLeaves += 1
            inactiveLeaves -= 1
          }
          supportedActiveNodes -= 1
        } else {
          if (leaf.stats.isActive) {
            leaf.deactivate()
            activeLeaves -= 1
            inactiveLeaves += 1
          }
        }
      }

    }

  }

  /**
   * Make a prediction on an unlabeled data point.
   *
   * @param point An unlabeled dat point to make a prediction on.
   * @return A two Tuple instance with the class prediction along with its confidence.
   */
  def predict(point: UnlabeledPoint): (Double, Double) = {
    try {
      val prediction: (Int, Double) = tree.predict(if (discrete) point.discreteVector else point.numericVector, "NaiveBayes")
      if (!classMap.containsValue(prediction._1))
        (Double.NaN, Double.NaN)
      else
        (classMap.inverse().get(prediction._1), prediction._2)
    } catch {
      case _: Throwable => (Double.NaN, Double.NaN)
    }
  }

}