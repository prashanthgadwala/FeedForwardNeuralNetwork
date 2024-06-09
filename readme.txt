Neural Network
In layer-oriented frameworks we typically have a neural network object which:
• is responsible for holding a graph of layers, whereas a "layer" represents a function (e.g. ReLU) or operation (e.g. convolution)
• weallowonlyextremelysimplegraphs • withalistoflayers
• andonlyonedatasource
• andonelossfunction
• is responsible to hold access to data
• has no explicit knowledge about the graph of layers it contains
• recursively calls forward on its layers passing the input-data
• recursively calls backward on its layers passing the error
• in our case stores the loss over iterations, while in other frameworks this is commonly separated into an optimizer class


