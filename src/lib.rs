mod aedat_core;

use numpy::convert::ToPyArray;
use numpy::ndarray::IntoDimension;
use numpy::prelude::*;
use numpy::Element;
use pyo3::prelude::*;

impl std::convert::From<aedat_core::ParseError> for pyo3::PyErr {
    fn from(error: aedat_core::ParseError) -> Self {
        pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyclass]
struct Decoder {
    decoder: aedat_core::Decoder,
    warned_unknown_frame_format: bool,
}

#[pymethods]
impl Decoder {
    #[new]
    fn new(path: &pyo3::Bound<'_, pyo3::types::PyAny>) -> Result<Self, pyo3::PyErr> {
        match python_path_to_string(path) {
            Ok(result) => match aedat_core::Decoder::new(result) {
                Ok(result) => Ok(Decoder {
                    decoder: result,
                    warned_unknown_frame_format: false,
                }),
                Err(error) => Err(pyo3::PyErr::from(error)),
            },
            Err(error) => Err(error),
        }
    }

    fn id_to_stream(
        &self,
        python: pyo3::prelude::Python,
    ) -> PyResult<pyo3::Py<pyo3::types::PyAny>> {
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

    fn __next__(mut shell: pyo3::PyRefMut<Self>) -> PyResult<Option<pyo3::Py<pyo3::types::PyAny>>> {
        loop {
            let packet = match shell.decoder.next() {
                Some(result) => match result {
                    Ok(result) => result,
                    Err(result) => return Err(pyo3::PyErr::from(result)),
                },
                None => return Ok(None),
            };
            let is_frame = matches!(
                shell
                    .decoder
                    .id_to_stream
                    .get(&packet.stream_id)
                    .map(|stream| stream.content),
                Some(aedat_core::StreamContent::Frame)
            );
            if is_frame {
                match aedat_core::frame_generated::size_prefixed_root_as_frame(&packet.buffer) {
                    Ok(frame) => {
                        if frame_format_info(frame.format()).is_none() {
                            if !shell.warned_unknown_frame_format {
                                let format_code = frame.format().0;
                                pyo3::Python::attach(|python| {
                                    let message = std::ffi::CString::new(format!(
                                        "skipping unknown frame format {format_code} (OpenCV type code); continuing with remaining packets"
                                    ))
                                    .expect("warning message");
                                    let category =
                                        python.get_type::<pyo3::exceptions::PyUserWarning>();
                                    let _ =
                                        pyo3::PyErr::warn(python, category.as_any(), &message, 1);
                                });
                                shell.warned_unknown_frame_format = true;
                            }
                            continue;
                        }
                    }
                    Err(_) => {
                        return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                            "the packet does not have a size prefix",
                        )))
                    }
                }
            }
            return pyo3::Python::attach(
                |python| -> PyResult<Option<pyo3::Py<pyo3::types::PyAny>>> {
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
                                    None,
                                    u64::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    1,
                                    "x",
                                    None,
                                    u16::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    2,
                                    "y",
                                    None,
                                    u16::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    3,
                                    "on",
                                    Some("p"),
                                    bool::get_dtype(python).num(),
                                );
                                let mut dtype: *mut numpy::npyffi::PyArray_Descr =
                                    std::ptr::null_mut();
                                if numpy::PY_ARRAY_API.PyArray_DescrConverter(
                                    python,
                                    dtype_as_list,
                                    &mut dtype,
                                ) < 0
                                {
                                    panic!("PyArray_DescrConverter failed");
                                }
                                pyo3::ffi::Py_DECREF(dtype_as_list);
                                let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                                    python,
                                    numpy::npyffi::get_type_object(
                                        python,
                                        numpy::npyffi::NpyTypes::PyArray_Type,
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
                                    )
                                        as *mut u8;
                                    let event = events.get(index as usize);
                                    let mut event_array = [0u8; 13];
                                    event_array[0..8]
                                        .copy_from_slice(&(event.t() as u64).to_ne_bytes());
                                    event_array[8..10]
                                        .copy_from_slice(&(event.x() as u16).to_ne_bytes());
                                    event_array[10..12]
                                        .copy_from_slice(&(event.y() as u16).to_ne_bytes());
                                    event_array[12] = if event.on() { 1 } else { 0 };
                                    std::ptr::copy(
                                        event_array.as_ptr(),
                                        event_cell,
                                        event_array.len(),
                                    );
                                }
                                pyo3::Bound::from_owned_ptr(python, array).unbind()
                            })?;
                        }
                        aedat_core::StreamContent::Frame => {
                            let frame =
                                match aedat_core::frame_generated::size_prefixed_root_as_frame(
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
                            let (format_name, channels, is_u16) =
                                match frame_format_info(frame.format()) {
                                    Some(info) => info,
                                    None => {
                                        return Err(pyo3::PyErr::from(aedat_core::ParseError::new(
                                            "unknown frame format",
                                        )))
                                    }
                                };
                            python_frame.set_item("format", format_name)?;
                            python_frame.set_item("width", frame.width())?;
                            python_frame.set_item("height", frame.height())?;
                            python_frame.set_item("offset_x", frame.offset_x())?;
                            python_frame.set_item("offset_y", frame.offset_y())?;
                            let height = frame.height() as usize;
                            let width = frame.width() as usize;
                            if channels == 1 {
                                let dimensions = [height, width].into_dimension();
                                if is_u16 {
                                    python_frame.set_item(
                                        "pixels",
                                        match frame.pixels() {
                                            Some(pixels) => le_u16_pixels(pixels.bytes())?
                                                .to_pyarray(python)
                                                .reshape(dimensions)?,
                                            None => numpy::array::PyArray2::<u16>::zeros(
                                                python, dimensions, false,
                                            ),
                                        },
                                    )?;
                                } else {
                                    python_frame.set_item(
                                        "pixels",
                                        match frame.pixels() {
                                            Some(pixels) => pixels
                                                .bytes()
                                                .to_pyarray(python)
                                                .reshape(dimensions)?,
                                            None => numpy::array::PyArray2::<u8>::zeros(
                                                python, dimensions, false,
                                            ),
                                        },
                                    )?;
                                }
                            } else {
                                let dimensions = [height, width, channels].into_dimension();
                                if is_u16 {
                                    python_frame.set_item(
                                        "pixels",
                                        match frame.pixels() {
                                            Some(pixels) => {
                                                let mut values = le_u16_pixels(pixels.bytes())?;
                                                swap_bgr_channel(&mut values, channels);
                                                values.to_pyarray(python).reshape(dimensions)?
                                            }
                                            None => numpy::array::PyArray3::<u16>::zeros(
                                                python, dimensions, false,
                                            ),
                                        },
                                    )?;
                                } else {
                                    python_frame.set_item(
                                        "pixels",
                                        match frame.pixels() {
                                            Some(pixels) => {
                                                let mut values = pixels.bytes().to_owned();
                                                swap_bgr_channel(&mut values, channels);
                                                values.to_pyarray(python).reshape(dimensions)?
                                            }
                                            None => numpy::array::PyArray3::<u8>::zeros(
                                                python, dimensions, false,
                                            ),
                                        },
                                    )?;
                                }
                            }
                            python_packet.set_item("frame", python_frame)?;
                        }
                        aedat_core::StreamContent::Imus => {
                            let imus =
                                match aedat_core::imus_generated::size_prefixed_root_as_imu_packet(
                                    &packet.buffer,
                                ) {
                                    Ok(result) => match result.elements() {
                                        Some(result) => result,
                                        None => {
                                            return Err(pyo3::PyErr::from(
                                                aedat_core::ParseError::new("empty events packet"),
                                            ))
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
                                    None,
                                    u64::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    1,
                                    "temperature",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    2,
                                    "accelerometer_x",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    3,
                                    "accelerometer_y",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    4,
                                    "accelerometer_z",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    5,
                                    "gyroscope_x",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    6,
                                    "gyroscope_y",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    7,
                                    "gyroscope_z",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    8,
                                    "magnetometer_x",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    9,
                                    "magnetometer_y",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    10,
                                    "magnetometer_z",
                                    None,
                                    f32::get_dtype(python).num(),
                                );
                                let mut dtype: *mut numpy::npyffi::PyArray_Descr =
                                    std::ptr::null_mut();
                                if numpy::PY_ARRAY_API.PyArray_DescrConverter(
                                    python,
                                    dtype_as_list,
                                    &mut dtype,
                                ) < 0
                                {
                                    panic!("PyArray_DescrConverter failed");
                                }
                                pyo3::ffi::Py_DECREF(dtype_as_list);
                                let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                                    python,
                                    numpy::npyffi::get_type_object(
                                        python,
                                        numpy::npyffi::NpyTypes::PyArray_Type,
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
                                    let mut imu_array = [0u8; 48];
                                    imu_array[0..8]
                                        .copy_from_slice(&(imu.t() as u64).to_ne_bytes());
                                    imu_array[8..12]
                                        .copy_from_slice(&(imu.temperature()).to_ne_bytes());
                                    imu_array[12..16]
                                        .copy_from_slice(&(imu.accelerometer_x()).to_ne_bytes());
                                    imu_array[16..20]
                                        .copy_from_slice(&(imu.accelerometer_y()).to_ne_bytes());
                                    imu_array[20..24]
                                        .copy_from_slice(&(imu.accelerometer_z()).to_ne_bytes());
                                    imu_array[24..28]
                                        .copy_from_slice(&(imu.gyroscope_x()).to_ne_bytes());
                                    imu_array[28..32]
                                        .copy_from_slice(&(imu.gyroscope_y()).to_ne_bytes());
                                    imu_array[32..36]
                                        .copy_from_slice(&(imu.gyroscope_z()).to_ne_bytes());
                                    imu_array[36..40]
                                        .copy_from_slice(&(imu.magnetometer_x()).to_ne_bytes());
                                    imu_array[40..44]
                                        .copy_from_slice(&(imu.magnetometer_y()).to_ne_bytes());
                                    imu_array[44..48]
                                        .copy_from_slice(&(imu.magnetometer_z()).to_ne_bytes());
                                    std::ptr::copy(imu_array.as_ptr(), imu_cell, imu_array.len());
                                    index += 1_isize;
                                }
                                pyo3::Bound::from_owned_ptr(python, array).unbind()
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
                                    None,
                                    u64::get_dtype(python).num(),
                                );
                                set_dtype_as_list_field(
                                    python,
                                    dtype_as_list,
                                    1,
                                    "source",
                                    None,
                                    u8::get_dtype(python).num(),
                                );
                                let mut dtype: *mut numpy::npyffi::PyArray_Descr =
                                    std::ptr::null_mut();
                                if numpy::PY_ARRAY_API.PyArray_DescrConverter(
                                    python,
                                    dtype_as_list,
                                    &mut dtype,
                                ) < 0
                                {
                                    panic!("PyArray_DescrConverter failed");
                                }
                                pyo3::ffi::Py_DECREF(dtype_as_list);
                                let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                                    python,
                                    numpy::npyffi::get_type_object(
                                        python,
                                        numpy::npyffi::NpyTypes::PyArray_Type,
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
                                    )
                                        as *mut u8;

                                    let mut trigger_array = [0u8; 9];
                                    trigger_array[0..8]
                                        .copy_from_slice(&(trigger.t() as u64).to_ne_bytes());
                                    use aedat_core::triggers_generated::TriggerSource;
                                    trigger_array[8] = match trigger.source() {
                                        TriggerSource::TimestampReset => 0_u8,
                                        TriggerSource::ExternalSignalRisingEdge => 1_u8,
                                        TriggerSource::ExternalSignalFallingEdge => 2_u8,
                                        TriggerSource::ExternalSignalPulse => 3_u8,
                                        TriggerSource::ExternalGeneratorRisingEdge => 4_u8,
                                        TriggerSource::ExternalGeneratorFallingEdge => 5_u8,
                                        TriggerSource::FrameBegin => 6_u8,
                                        TriggerSource::FrameEnd => 7_u8,
                                        TriggerSource::ExposureBegin => 8_u8,
                                        TriggerSource::ExposureEnd => 9_u8,
                                        _ => {
                                            return Err(pyo3::PyErr::from(
                                                aedat_core::ParseError::new(
                                                    "unknown trigger source",
                                                ),
                                            ))
                                        }
                                    };
                                    std::ptr::copy(
                                        trigger_array.as_ptr(),
                                        trigger_cell,
                                        trigger_array.len(),
                                    );
                                    index += 1_isize;
                                }
                                pyo3::Bound::from_owned_ptr(python, array).unbind()
                            })?;
                        }
                    }
                    Ok(Some(python_packet.into()))
                },
            );
        }
    }
}

fn frame_format_info(
    format: aedat_core::frame_generated::FrameFormat,
) -> Option<(&'static str, usize, bool)> {
    match format {
        aedat_core::frame_generated::FrameFormat::Gray => Some(("L", 1, false)),
        aedat_core::frame_generated::FrameFormat::Gray16 => Some(("I;16", 1, true)),
        aedat_core::frame_generated::FrameFormat::Bgr => Some(("RGB", 3, false)),
        aedat_core::frame_generated::FrameFormat::Bgr16 => Some(("RGB", 3, true)),
        aedat_core::frame_generated::FrameFormat::Bgra => Some(("RGBA", 4, false)),
        aedat_core::frame_generated::FrameFormat::Bgra16 => Some(("RGBA", 4, true)),
        _ => None,
    }
}

fn le_u16_pixels(bytes: &[u8]) -> Result<Vec<u16>, aedat_core::ParseError> {
    if bytes.len() % 2 != 0 {
        return Err(aedat_core::ParseError::new(
            "16-bit frame pixel buffer length is not a multiple of 2",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn swap_bgr_channel<T>(pixels: &mut [T], channels: usize) {
    if channels < 3 {
        return;
    }
    for index in 0..(pixels.len() / channels) {
        pixels.swap(index * channels, index * channels + 2);
    }
}

unsafe fn set_dtype_as_list_field(
    python: pyo3::Python,
    list: *mut pyo3::ffi::PyObject,
    index: i32,
    name: &str,
    title: Option<&str>,
    numpy_type: core::ffi::c_int,
) {
    let tuple = pyo3::ffi::PyTuple_New(2);
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        0 as pyo3::ffi::Py_ssize_t,
        match title {
            Some(title) => {
                let tuple = pyo3::ffi::PyTuple_New(2);
                if pyo3::ffi::PyTuple_SetItem(
                    tuple,
                    0 as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyUnicode_FromStringAndSize(
                        title.as_ptr() as *const core::ffi::c_char,
                        title.len() as pyo3::ffi::Py_ssize_t,
                    ),
                ) < 0
                {
                    panic!("PyTuple_SetItem 0 failed");
                }
                if pyo3::ffi::PyTuple_SetItem(
                    tuple,
                    1 as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyUnicode_FromStringAndSize(
                        name.as_ptr() as *const core::ffi::c_char,
                        name.len() as pyo3::ffi::Py_ssize_t,
                    ),
                ) < 0
                {
                    panic!("PyTuple_SetItem 1 failed");
                }
                tuple
            }
            None => pyo3::ffi::PyUnicode_FromStringAndSize(
                name.as_ptr() as *const core::ffi::c_char,
                name.len() as pyo3::ffi::Py_ssize_t,
            ),
        },
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

fn python_path_to_string(path: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<String> {
    if let Ok(result) = path.cast::<pyo3::types::PyString>() {
        return Ok(result.to_string());
    }
    if let Ok(result) = path.cast::<pyo3::types::PyBytes>() {
        return Ok(result.to_string());
    }
    let fspath_result = path.call_method0("__fspath__")?;
    {
        let fspath_as_string: Result<
            &pyo3::Bound<'_, pyo3::types::PyString>,
            pyo3::CastError<'_, '_>,
        > = fspath_result.cast();
        if let Ok(result) = fspath_as_string {
            return Ok(result.to_string());
        }
    }
    let fspath_as_bytes: &pyo3::Bound<'_, pyo3::types::PyBytes> = fspath_result
        .cast()
        .map_err(|__fspath__| pyo3::exceptions::PyTypeError::new_err("path must be a string, bytes, or an object with an __fspath__ method (such as pathlib.Path"))?;
    Ok(fspath_as_bytes.to_string())
}

#[pymodule(gil_used = false)]
fn aedat(
    _python: pyo3::prelude::Python,
    module: &pyo3::Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_class::<Decoder>()?;
    Ok(())
}
