pub error SubstrateError =
    OutOfMemory
    DeviceLost
    CompileError(msg: str)
    ShapeMismatch(msg: str)
    DTypeMismatch(msg: str)
    InvalidView(msg: str)
    StreamError(msg: str)
    GridExceedsDevice(msg: str)
    BroadcastWriteViolation
    Unsupported(msg: str)
