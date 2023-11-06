extern crate ndarray;
use ndarray::IntoDimension;
extern crate numpy;
use numpy::convert::ToPyArray;
use numpy::Element;
extern crate pyo3;
use pyo3::prelude::*;
mod aedat_core;

impl std::convert::From<aedat_core::ParseError> for pyo3::PyErr {
    fn from(error: aedat_core::ParseError) -> Self {
        pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyclass]
struct Decoder {
    decoder: aedat_core::Decoder,
}

#[pymethods]
impl Decoder {
    #[new]
    fn new(path: &pyo3::types::PyAny) -> Result<Self, pyo3::PyErr> {
        pyo3::Python::with_gil(|python| -> Result<Self, pyo3::PyErr> {
            match python_path_to_string(python, path) {
                Ok(result) => match aedat_core::Decoder::new(result) {
                    Ok(result) => Ok(Decoder { decoder: result }),
                    Err(error) => Err(pyo3::PyErr::from(error)),
                },
                Err(error) => Err(error),
            }
        })
    }

    fn id_to_stream(&self, python: pyo3::prelude::Python) -> PyResult<PyObject> {
        let python_id_to_stream = pyo3::types::PyDict::new(python);
        for (id, stream) in self.decoder.id_to_stream.iter() {
            let python_stream = pyo3::types::PyDict::new(python);
            match stream.content {
                aedat_core::StreamContent::Events => {
                    python_stream.set_item("type", "events")?;
                    python_stream.set_item("width", stream.width)?;
                    python_stream.set_item("height", stream.height)?;
                }
                aedat_core::StreamContent::Frame => {
                    python_stream.set_item("type", "frame")?;
                    python_stream.set_item("width", stream.width)?;
                    python_stream.set_item("height", stream.height)?;
                }
                aedat_core::StreamContent::Imus => python_stream.set_item("type", "imus")?,
                aedat_core::StreamContent::Triggers => {
                    python_stream.set_item("type", "triggers")?
                }
            }
            python_id_to_stream.set_item(id, python_stream)?;
        }
        Ok(python_id_to_stream.into())
    }

    fn __iter__(shell: pyo3::PyRefMut<Self>) -> PyResult<pyo3::prelude::Py<Decoder>> {
        Ok(shell.into())
    }

    fn __next__(mut shell: pyo3::PyRefMut<Self>) -> PyResult<Option<PyObject>> {
        let packet = match shell.decoder.next() {
            Some(result) => match result {
                Ok(result) => result,
                Err(result) => return Err(pyo3::PyErr::from(result)),
            },
            None => return Ok(None),
        };
        pyo3::Python::with_gil(|python| -> PyResult<Option<PyObject>> {
            let python_packet = pyo3::types::PyDict::new(python);
            python_packet.set_item("stream_id", packet.stream_id)?;
            match shell
                .decoder
                .id_to_stream
                .get(&packet.stream_id)
                .unwrap()
                .content
            {
                aedat_core::StreamContent::Events => {
                    let events =
                        match aedat_core::events_generated::size_prefixed_root_as_event_packet(
                            &packet.buffer,
                        ) {
                            Ok(result) => match result.elements() {
                                Some(result) => result,
                                None => {
                                    return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                        "empty events packet",
                                    )))
                                }
                            },
                            Err(_) => {
                                return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                    "the packet does not have a size prefix",
                                )))
                            }
                        };
                    let mut length = events.len() as numpy::npyffi::npy_intp;
                    python_packet.set_item("events", unsafe {
                        let dtype_as_list = pyo3::ffi::PyList_New(4_isize);
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            0,
                            "t",
                            u64::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            1,
                            "x",
                            u16::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            2,
                            "y",
                            u16::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            3,
                            "on",
                            bool::get_dtype(python).num(),
                        );
                        let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
                        if numpy::PY_ARRAY_API.PyArray_DescrConverter(
                            python,
                            dtype_as_list,
                            &mut dtype,
                        ) < 0
                        {
                            panic!("PyArray_DescrConverter failed");
                        }
                        let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                            python,
                            numpy::PY_ARRAY_API.get_type_object(
                                python,
                                numpy::npyffi::array::NpyTypes::PyArray_Type,
                            ),
                            dtype,
                            1_i32,
                            &mut length as *mut numpy::npyffi::npy_intp,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            0_i32,
                            std::ptr::null_mut(),
                        );
                        for mut index in 0_isize..length {
                            let event_cell = numpy::PY_ARRAY_API.PyArray_GetPtr(
                                python,
                                array as *mut numpy::npyffi::PyArrayObject,
                                &mut index as *mut numpy::npyffi::npy_intp,
                            ) as *mut u8;
                            let event = events.get(index as usize);
                            *(event_cell.offset(0) as *mut u64) = event.t() as u64;
                            *(event_cell.offset(8) as *mut u16) = event.x() as u16;
                            *(event_cell.offset(10) as *mut u16) = event.y() as u16;
                            *event_cell.offset(12) = u8::from(event.on());
                        }
                        PyObject::from_owned_ptr(python, array)
                    })?;
                }
                aedat_core::StreamContent::Frame => {
                    let frame = match aedat_core::frame_generated::size_prefixed_root_as_frame(
                        &packet.buffer,
                    ) {
                        Ok(result) => result,
                        Err(_) => {
                            return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                "the packet does not have a size prefix",
                            )))
                        }
                    };
                    let python_frame = pyo3::types::PyDict::new(python);
                    python_frame.set_item("t", frame.t())?;
                    python_frame.set_item("begin_t", frame.begin_t())?;
                    python_frame.set_item("end_t", frame.end_t())?;
                    python_frame.set_item("exposure_begin_t", frame.exposure_begin_t())?;
                    python_frame.set_item("exposure_end_t", frame.exposure_end_t())?;
                    python_frame.set_item(
                        "format",
                        match frame.format() {
                            aedat_core::frame_generated::FrameFormat::Gray => "L",
                            aedat_core::frame_generated::FrameFormat::Bgr => "RGB",
                            aedat_core::frame_generated::FrameFormat::Bgra => "RGBA",
                            _ => {
                                return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                    "unknown frame format",
                                )))
                            }
                        },
                    )?;
                    python_frame.set_item("width", frame.width())?;
                    python_frame.set_item("height", frame.height())?;
                    python_frame.set_item("offset_x", frame.offset_x())?;
                    python_frame.set_item("offset_y", frame.offset_y())?;
                    match frame.format() {
                        aedat_core::frame_generated::FrameFormat::Gray => {
                            let dimensions =
                                [frame.height() as usize, frame.width() as usize].into_dimension();
                            python_frame.set_item(
                                "pixels",
                                match frame.pixels() {
                                    Some(result) => {
                                        result.bytes().to_pyarray(python).reshape(dimensions)?
                                    }
                                    None => numpy::array::PyArray2::<u8>::zeros(
                                        python, dimensions, false,
                                    ),
                                },
                            )?;
                        }
                        aedat_core::frame_generated::FrameFormat::Bgr
                        | aedat_core::frame_generated::FrameFormat::Bgra => {
                            let channels = if frame.format()
                                == aedat_core::frame_generated::FrameFormat::Bgr
                            {
                                3_usize
                            } else {
                                4_usize
                            };
                            let dimensions =
                                [frame.height() as usize, frame.width() as usize, channels]
                                    .into_dimension();
                            python_frame.set_item(
                                "pixels",
                                match frame.pixels() {
                                    Some(result) => {
                                        let mut pixels = result.bytes().to_owned();
                                        for index in 0..(pixels.len() / channels) {
                                            pixels.swap(index * channels, index * channels + 2);
                                        }
                                        pixels.to_pyarray(python).reshape(dimensions)?
                                    }
                                    None => numpy::array::PyArray3::<u8>::zeros(
                                        python, dimensions, false,
                                    ),
                                },
                            )?;
                        }
                        _ => {
                            return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                "unknown frame format",
                            )))
                        }
                    }
                    python_packet.set_item("frame", python_frame)?;
                }
                aedat_core::StreamContent::Imus => {
                    let imus = match aedat_core::imus_generated::size_prefixed_root_as_imu_packet(
                        &packet.buffer,
                    ) {
                        Ok(result) => match result.elements() {
                            Some(result) => result,
                            None => {
                                return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                    "empty events packet",
                                )))
                            }
                        },
                        Err(_) => {
                            return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                "the packet does not have a size prefix",
                            )))
                        }
                    };
                    let mut length = imus.len() as numpy::npyffi::npy_intp;
                    python_packet.set_item("imus", unsafe {
                        let dtype_as_list = pyo3::ffi::PyList_New(11_isize);
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            0,
                            "t",
                            u64::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            1,
                            "temperature",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            2,
                            "accelerometer_x",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            3,
                            "accelerometer_y",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            4,
                            "accelerometer_z",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            5,
                            "gyroscope_x",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            6,
                            "gyroscope_y",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            7,
                            "gyroscope_z",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            8,
                            "magnetometer_x",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            9,
                            "magnetometer_y",
                            f32::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            10,
                            "magnetometer_z",
                            f32::get_dtype(python).num(),
                        );
                        let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
                        if numpy::PY_ARRAY_API.PyArray_DescrConverter(
                            python,
                            dtype_as_list,
                            &mut dtype,
                        ) < 0
                        {
                            panic!("PyArray_DescrConverter failed");
                        }
                        let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                            python,
                            numpy::PY_ARRAY_API.get_type_object(
                                python,
                                numpy::npyffi::array::NpyTypes::PyArray_Type,
                            ),
                            dtype,
                            1_i32,
                            &mut length as *mut numpy::npyffi::npy_intp,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            0_i32,
                            std::ptr::null_mut(),
                        );
                        let mut index = 0_isize;
                        for imu in imus {
                            let imu_cell = numpy::PY_ARRAY_API.PyArray_GetPtr(
                                python,
                                array as *mut numpy::npyffi::PyArrayObject,
                                &mut index as *mut numpy::npyffi::npy_intp,
                            ) as *mut u8;
                            *(imu_cell.offset(0) as *mut u64) = imu.t() as u64;
                            *(imu_cell.offset(8) as *mut f32) = imu.temperature();
                            *(imu_cell.offset(12) as *mut f32) = imu.accelerometer_x();
                            *(imu_cell.offset(16) as *mut f32) = imu.accelerometer_y();
                            *(imu_cell.offset(20) as *mut f32) = imu.accelerometer_z();
                            *(imu_cell.offset(24) as *mut f32) = imu.gyroscope_x();
                            *(imu_cell.offset(28) as *mut f32) = imu.gyroscope_y();
                            *(imu_cell.offset(32) as *mut f32) = imu.gyroscope_z();
                            *(imu_cell.offset(36) as *mut f32) = imu.magnetometer_x();
                            *(imu_cell.offset(40) as *mut f32) = imu.magnetometer_y();
                            *(imu_cell.offset(44) as *mut f32) = imu.magnetometer_z();
                            index += 1_isize;
                        }
                        PyObject::from_owned_ptr(python, array)
                    })?;
                }
                aedat_core::StreamContent::Triggers => {
                    let triggers =
                        match aedat_core::triggers_generated::size_prefixed_root_as_trigger_packet(
                            &packet.buffer,
                        ) {
                            Ok(result) => match result.elements() {
                                Some(result) => result,
                                None => {
                                    return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                        "empty events packet",
                                    )))
                                }
                            },
                            Err(_) => {
                                return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                    "the packet does not have a size prefix",
                                )))
                            }
                        };
                    let mut length = triggers.len() as numpy::npyffi::npy_intp;
                    python_packet.set_item("triggers", unsafe {
                        let dtype_as_list = pyo3::ffi::PyList_New(2_isize);
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            0,
                            "t",
                            u64::get_dtype(python).num(),
                        );
                        set_dtype_as_list_field(
                            python,
                            dtype_as_list,
                            1,
                            "source",
                            u8::get_dtype(python).num(),
                        );
                        let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
                        if numpy::PY_ARRAY_API.PyArray_DescrConverter(
                            python,
                            dtype_as_list,
                            &mut dtype,
                        ) < 0
                        {
                            panic!("PyArray_DescrConverter failed");
                        }
                        let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                            python,
                            numpy::PY_ARRAY_API.get_type_object(
                                python,
                                numpy::npyffi::array::NpyTypes::PyArray_Type,
                            ),
                            dtype,
                            1_i32,
                            &mut length as *mut numpy::npyffi::npy_intp,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            0_i32,
                            std::ptr::null_mut(),
                        );
                        let mut index = 0_isize;
                        for trigger in triggers {
                            let trigger_cell = numpy::PY_ARRAY_API.PyArray_GetPtr(
                                python,
                                array as *mut numpy::npyffi::PyArrayObject,
                                &mut index as *mut numpy::npyffi::npy_intp,
                            ) as *mut u8;
                            *(trigger_cell.offset(0) as *mut u64) = trigger.t() as u64;
                            *trigger_cell.offset(8) = match trigger.source() {
                                aedat_core::triggers_generated::TriggerSource::TimestampReset => 0_u8,
                                aedat_core::triggers_generated::TriggerSource::ExternalSignalRisingEdge => 1_u8,
                                aedat_core::triggers_generated::TriggerSource::ExternalSignalFallingEdge => {
                                    2_u8
                                }
                                aedat_core::triggers_generated::TriggerSource::ExternalSignalPulse => 3_u8,
                                aedat_core::triggers_generated::TriggerSource::ExternalGeneratorRisingEdge => {
                                    4_u8
                                }
                                aedat_core::triggers_generated::TriggerSource::ExternalGeneratorFallingEdge => {
                                    5_u8
                                }
                                aedat_core::triggers_generated::TriggerSource::FrameBegin => 6_u8,
                                aedat_core::triggers_generated::TriggerSource::FrameEnd => 7_u8,
                                aedat_core::triggers_generated::TriggerSource::ExposureBegin => 8_u8,
                                aedat_core::triggers_generated::TriggerSource::ExposureEnd => 9_u8,
                                _ => {
                                    return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                        "unknown trigger source",
                                    )))
                                }
                            };
                            index += 1_isize;
                        }
                        PyObject::from_owned_ptr(python, array)
                    })?;
                }
            }
            Ok(Some(python_packet.into()))
        })
    }
}

unsafe fn set_dtype_as_list_field(
    python: pyo3::Python,
    list: *mut pyo3::ffi::PyObject,
    index: i32,
    name: &str,
    numpy_type: core::ffi::c_int,
) {
    let tuple = pyo3::ffi::PyTuple_New(2);
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        0 as pyo3::ffi::Py_ssize_t,
        pyo3::ffi::PyUnicode_FromStringAndSize(
            name.as_ptr() as *const core::ffi::c_char,
            name.len() as pyo3::ffi::Py_ssize_t,
        ),
    ) < 0
    {
        panic!("PyTuple_SetItem 0 failed");
    }
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        1 as pyo3::ffi::Py_ssize_t,
        numpy::PY_ARRAY_API.PyArray_TypeObjectFromType(python, numpy_type),
    ) < 0
    {
        panic!("PyTuple_SetItem 1 failed");
    }
    if pyo3::ffi::PyList_SetItem(list, index as pyo3::ffi::Py_ssize_t, tuple) < 0 {
        panic!("PyList_SetItem failed");
    }
}

fn python_path_to_string(
    python: pyo3::prelude::Python,
    path: &pyo3::types::PyAny,
) -> PyResult<String> {
    if let Ok(result) = path.downcast::<pyo3::types::PyString>() {
        return Ok(result.to_string());
    }
    if let Ok(result) = path.downcast::<pyo3::types::PyBytes>() {
        return Ok(result.to_string());
    }
    let fspath_result = path.to_object(python).call_method0(python, "__fspath__")?;
    {
        let fspath_as_string: PyResult<&pyo3::types::PyString> = fspath_result.extract(python);
        if let Ok(result) = fspath_as_string {
            return Ok(result.to_string());
        }
    }
    let fspath_as_bytes: &pyo3::types::PyBytes = fspath_result.extract(python)?;
    Ok(fspath_as_bytes.to_string())
}

#[pymodule]
fn aedat(_python: pyo3::prelude::Python, module: &pyo3::prelude::PyModule) -> PyResult<()> {
    module.add_class::<Decoder>()?;

    Ok(())
}
