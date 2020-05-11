#![feature(specialization, const_fn)]
extern crate ndarray;
use ndarray::IntoDimension;
extern crate numpy;
use numpy::convert::ToPyArray;
use numpy::types::TypeNum;
extern crate pyo3;
use pyo3::prelude::PyResult;
use pyo3::{PyObject, ToPyObject};
mod aedat;
#[allow(dead_code, unused_imports)]
#[path = "./events_generated.rs"]
mod events_generated;
#[allow(dead_code, unused_imports)]
#[path = "./frame_generated.rs"]
mod frame_generated;
#[allow(dead_code, unused_imports)]
#[path = "./imus_generated.rs"]
mod imus_generated;
#[allow(dead_code, unused_imports)]
#[path = "./triggers_generated.rs"]
mod triggers_generated;

impl std::convert::From<aedat::ParseError> for pyo3::PyErr {
    fn from(error: aedat::ParseError) -> Self {
        pyo3::PyErr::new::<pyo3::exceptions::RuntimeError, _>(error.to_string())
    }
}

#[pyo3::prelude::pyclass]
struct Decoder {
    decoder: aedat::Decoder,
}

#[pyo3::prelude::pymethods]
impl Decoder {
    #[new]
    fn new(object: &pyo3::PyRawObject, path: &pyo3::types::PyAny) -> PyResult<()> {
        let gil = pyo3::Python::acquire_gil();
        let python = gil.python();
        match python_path_to_string(python, path) {
            Ok(result) => match aedat::Decoder::new(result) {
                Ok(result) => object.init(Decoder { decoder: result }),
                Err(error) => return Err(pyo3::PyErr::from(error)),
            },
            Err(error) => return Err(error),
        };
        Ok(())
    }

    fn id_to_stream(&self, python: pyo3::prelude::Python) -> PyResult<PyObject> {
        let python_id_to_stream = pyo3::types::PyDict::new(python);
        for (id, stream) in self.decoder.id_to_stream.iter() {
            let python_stream = pyo3::types::PyDict::new(python);
            match stream.content {
                aedat::StreamContent::Events => {
                    python_stream.set_item("type", "events")?;
                    python_stream.set_item("width", stream.width)?;
                    python_stream.set_item("height", stream.height)?;
                }
                aedat::StreamContent::Frame => {
                    python_stream.set_item("type", "frame")?;
                    python_stream.set_item("width", stream.width)?;
                    python_stream.set_item("height", stream.height)?;
                }
                aedat::StreamContent::Imus => python_stream.set_item("type", "imus")?,
                aedat::StreamContent::Triggers => python_stream.set_item("type", "triggers")?,
            }
            python_id_to_stream.set_item(id, python_stream)?;
        }
        Ok(python_id_to_stream.into())
    }
}

unsafe fn set_dtype_as_list_field(
    list: *mut pyo3::ffi::PyObject,
    index: i32,
    name: &str,
    numpy_type: i32,
) {
    let tuple = pyo3::ffi::PyTuple_New(2);
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        0 as pyo3::ffi::Py_ssize_t,
        pyo3::ffi::PyUnicode_FromStringAndSize(
            name.as_ptr() as *const std::os::raw::c_char,
            name.len() as pyo3::ffi::Py_ssize_t,
        ),
    ) < 0
    {
        panic!("PyTuple_SetItem 0 failed");
    }
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        1 as pyo3::ffi::Py_ssize_t,
        numpy::PY_ARRAY_API.PyArray_TypeObjectFromType(numpy_type),
    ) < 0
    {
        panic!("PyTuple_SetItem 1 failed");
    }
    if pyo3::ffi::PyList_SetItem(list, index as pyo3::ffi::Py_ssize_t, tuple) < 0 {
        panic!("PyList_SetItem failed");
    }
}

#[pyo3::prelude::pyproto]
impl pyo3::PyIterProtocol for Decoder {
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
        let gil = pyo3::Python::acquire_gil();
        let python = gil.python();
        let python_packet = pyo3::types::PyDict::new(python);
        python_packet.set_item("stream_id", packet.stream_id)?;
        match shell
            .decoder
            .id_to_stream
            .get(&packet.stream_id)
            .unwrap()
            .content
        {
            aedat::StreamContent::Events => {
                let events =
                    match events_generated::get_size_prefixed_root_as_event_packet(&packet.buffer)
                        .elements()
                    {
                        Some(result) => result,
                        None => {
                            return Err(pyo3::PyErr::from(aedat::ParseError::new(
                                "empty events packet",
                            )))
                        }
                    };
                let mut length = events.len() as numpy::npyffi::npy_intp;
                python_packet.set_item("events", unsafe {
                    let dtype_as_list = pyo3::ffi::PyList_New(4 as pyo3::ffi::Py_ssize_t);
                    set_dtype_as_list_field(dtype_as_list, 0, "t", u64::typenum_default());
                    set_dtype_as_list_field(dtype_as_list, 1, "x", u16::typenum_default());
                    set_dtype_as_list_field(dtype_as_list, 2, "y", u16::typenum_default());
                    set_dtype_as_list_field(dtype_as_list, 3, "on", bool::typenum_default());
                    let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
                    if numpy::PY_ARRAY_API.PyArray_DescrConverter(dtype_as_list, &mut dtype) < 0 {
                        panic!("PyArray_DescrConverter failed");
                    }
                    let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                        numpy::PY_ARRAY_API.get_type_object(numpy::npyffi::ArrayType::PyArray_Type),
                        dtype,
                        1 as std::os::raw::c_int,
                        &mut length as *mut numpy::npyffi::npy_intp,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0 as std::os::raw::c_int,
                        std::ptr::null_mut(),
                    );
                    for mut index in { 0 as numpy::npyffi::npy_intp }..length {
                        let event_cell = numpy::PY_ARRAY_API.PyArray_GetPtr(
                            array as *mut numpy::npyffi::PyArrayObject,
                            &mut index as *mut numpy::npyffi::npy_intp,
                        ) as *mut u8;
                        let event = events[index as usize];
                        *(event_cell.offset(0) as *mut u64) = event.t() as u64;
                        *(event_cell.offset(8) as *mut u16) = event.x() as u16;
                        *(event_cell.offset(10) as *mut u16) = event.y() as u16;
                        *(event_cell.offset(12) as *mut u8) = if event.on() { 1u8 } else { 0u8 };
                    }
                    PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                })?;
            }
            aedat::StreamContent::Frame => {
                let frame = frame_generated::get_size_prefixed_root_as_frame(&packet.buffer);
                let python_frame = pyo3::types::PyDict::new(python);
                python_frame.set_item("t", frame.t())?;
                python_frame.set_item("begin_t", frame.begin_t())?;
                python_frame.set_item("end_t", frame.end_t())?;
                python_frame.set_item("exposure_begin_t", frame.exposure_begin_t())?;
                python_frame.set_item("exposure_end_t", frame.exposure_end_t())?;
                python_frame.set_item(
                    "format",
                    match frame.format() {
                        frame_generated::FrameFormat::Gray => "Gray",
                        frame_generated::FrameFormat::Bgr => "BGR",
                        frame_generated::FrameFormat::Bgra => "BGRA",
                    },
                )?;
                python_frame.set_item("width", frame.width())?;
                python_frame.set_item("height", frame.height())?;
                python_frame.set_item("offset_x", frame.offset_x())?;
                python_frame.set_item("offset_y", frame.offset_y())?;
                let dimensions = [frame.height() as usize, frame.width() as usize].into_dimension();
                python_frame.set_item(
                    "pixels",
                    match frame.pixels() {
                        Some(result) => result.to_pyarray(python).reshape(dimensions)?,
                        None => numpy::array::PyArray2::<u8>::zeros(python, dimensions, false),
                    },
                )?;
                python_packet.set_item("frame", python_frame)?;
            }
            aedat::StreamContent::Imus => {
                let imus =
                    match imus_generated::get_size_prefixed_root_as_imu_packet(&packet.buffer)
                        .elements()
                    {
                        Some(result) => result,
                        None => {
                            return Err(pyo3::PyErr::from(aedat::ParseError::new(
                                "empty buffer packet",
                            )))
                        }
                    };
                let mut length = imus.len() as numpy::npyffi::npy_intp;
                python_packet.set_item("imus", unsafe {
                    let dtype_as_list = pyo3::ffi::PyList_New(11 as pyo3::ffi::Py_ssize_t);
                    set_dtype_as_list_field(dtype_as_list, 0, "t", u64::typenum_default());
                    set_dtype_as_list_field(
                        dtype_as_list,
                        1,
                        "temperature",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        2,
                        "accelerometer_x",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        3,
                        "accelerometer_y",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        4,
                        "accelerometer_z",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        5,
                        "gyroscope_x",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        6,
                        "gyroscope_y",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        7,
                        "gyroscope_z",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        8,
                        "magnetometer_x",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        9,
                        "magnetometer_y",
                        f32::typenum_default(),
                    );
                    set_dtype_as_list_field(
                        dtype_as_list,
                        10,
                        "magnetometer_z",
                        f32::typenum_default(),
                    );
                    let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
                    if numpy::PY_ARRAY_API.PyArray_DescrConverter(dtype_as_list, &mut dtype) < 0 {
                        panic!("PyArray_DescrConverter failed");
                    }
                    let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                        numpy::PY_ARRAY_API.get_type_object(numpy::npyffi::ArrayType::PyArray_Type),
                        dtype,
                        1 as std::os::raw::c_int,
                        &mut length as *mut numpy::npyffi::npy_intp,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0 as std::os::raw::c_int,
                        std::ptr::null_mut(),
                    );
                    let mut index = 0 as numpy::npyffi::npy_intp;
                    for imu in imus {
                        let imu_cell = numpy::PY_ARRAY_API.PyArray_GetPtr(
                            array as *mut numpy::npyffi::PyArrayObject,
                            &mut index as *mut numpy::npyffi::npy_intp,
                        ) as *mut u8;
                        *(imu_cell.offset(0) as *mut u64) = imu.t() as u64;
                        *(imu_cell.offset(8) as *mut f32) = imu.temperature();
                        *(imu_cell.offset(12) as *mut f32) = imu.accelerometer_x();
                        *(imu_cell.offset(16) as *mut f32) = imu.accelerometer_y();
                        *(imu_cell.offset(20) as *mut f32) = imu.accelerometer_z();
                        *(imu_cell.offset(12) as *mut f32) = imu.gyroscope_x();
                        *(imu_cell.offset(16) as *mut f32) = imu.gyroscope_y();
                        *(imu_cell.offset(20) as *mut f32) = imu.gyroscope_z();
                        *(imu_cell.offset(12) as *mut f32) = imu.magnetometer_x();
                        *(imu_cell.offset(16) as *mut f32) = imu.magnetometer_y();
                        *(imu_cell.offset(20) as *mut f32) = imu.magnetometer_z();
                        index += 1 as numpy::npyffi::npy_intp;
                    }
                    PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                })?;
            }
            aedat::StreamContent::Triggers => {
                let triggers = match triggers_generated::get_size_prefixed_root_as_trigger_packet(
                    &packet.buffer,
                )
                .elements()
                {
                    Some(result) => result,
                    None => {
                        return Err(pyo3::PyErr::from(aedat::ParseError::new(
                            "empty triggers packet",
                        )))
                    }
                };
                let mut length = triggers.len() as numpy::npyffi::npy_intp;
                python_packet.set_item("triggers", unsafe {
                    let dtype_as_list = pyo3::ffi::PyList_New(2 as pyo3::ffi::Py_ssize_t);
                    set_dtype_as_list_field(dtype_as_list, 0, "t", u64::typenum_default());
                    set_dtype_as_list_field(dtype_as_list, 1, "source", u8::typenum_default());
                    let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
                    if numpy::PY_ARRAY_API.PyArray_DescrConverter(dtype_as_list, &mut dtype) < 0 {
                        panic!("PyArray_DescrConverter failed");
                    }
                    let array = numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                        numpy::PY_ARRAY_API.get_type_object(numpy::npyffi::ArrayType::PyArray_Type),
                        dtype,
                        1 as std::os::raw::c_int,
                        &mut length as *mut numpy::npyffi::npy_intp,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0 as std::os::raw::c_int,
                        std::ptr::null_mut(),
                    );
                    let mut index = 0 as numpy::npyffi::npy_intp;
                    for trigger in triggers {
                        let trigger_cell = numpy::PY_ARRAY_API.PyArray_GetPtr(
                            array as *mut numpy::npyffi::PyArrayObject,
                            &mut index as *mut numpy::npyffi::npy_intp,
                        ) as *mut u8;
                        *(trigger_cell.offset(0) as *mut u64) = trigger.t() as u64;
                        *(trigger_cell.offset(8) as *mut u8) = trigger.source() as u8;
                        index += 1 as numpy::npyffi::npy_intp;
                    }
                    PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                })?;
            }
        };
        Ok(Some(python_packet.into()))
    }
}

fn python_path_to_string(
    python: pyo3::prelude::Python,
    path: &pyo3::types::PyAny,
) -> PyResult<String> {
    if let Ok(result) = path.downcast_ref::<pyo3::types::PyString>() {
        if let Ok(result) = result.to_string() {
            return Ok(result.to_string());
        }
    }
    if let Ok(result) = path.downcast_ref::<pyo3::types::PyBytes>() {
        return Ok(result.to_string());
    }
    let fspath_result = path.to_object(python).call_method0(python, "__fspath__")?;
    {
        let fspath_as_string: PyResult<&pyo3::types::PyString> = fspath_result.extract(python);
        if let Ok(result) = fspath_as_string {
            if let Ok(result) = result.to_string() {
                return Ok(result.to_string());
            }
        }
    }
    let fspath_as_bytes: &pyo3::types::PyBytes = fspath_result.extract(python)?;
    Ok(fspath_as_bytes.to_string())
}

#[pyo3::prelude::pymodule]
fn aedat(_python: pyo3::prelude::Python, module: &pyo3::prelude::PyModule) -> PyResult<()> {
    module.add_class::<Decoder>()?;

    Ok(())
}
