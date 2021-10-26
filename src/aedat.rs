use std::io::Read;

#[allow(dead_code, unused_imports)]
#[path = "./ioheader_generated.rs"]
mod ioheader_generated;

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

const MAGIC_NUMBER: &str = "#!AER-DAT4.0\r\n";

#[derive(Debug)]
pub struct ParseError {
    message: String,
}

impl ParseError {
    pub fn new(message: &str) -> ParseError {
        ParseError {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::convert::From<std::io::Error> for ParseError {
    fn from(error: std::io::Error) -> Self {
        ParseError {
            message: error.to_string(),
        }
    }
}

impl std::convert::From<flatbuffers::InvalidFlatbuffer> for ParseError {
    fn from(error: flatbuffers::InvalidFlatbuffer) -> Self {
        ParseError {
            message: error.to_string(),
        }
    }
}

impl std::convert::From<std::str::Utf8Error> for ParseError {
    fn from(error: std::str::Utf8Error) -> Self {
        ParseError {
            message: error.to_string(),
        }
    }
}

impl std::convert::From<roxmltree::Error> for ParseError {
    fn from(error: roxmltree::Error) -> Self {
        ParseError {
            message: error.to_string(),
        }
    }
}

impl std::convert::From<std::num::ParseIntError> for ParseError {
    fn from(error: std::num::ParseIntError) -> Self {
        ParseError {
            message: error.to_string(),
        }
    }
}

pub enum StreamContent {
    Events,
    Frame,
    Imus,
    Triggers,
}

impl StreamContent {
    fn from(identifier: &str) -> Result<Self, ParseError> {
        match identifier {
            "EVTS" => Ok(StreamContent::Events),
            "FRME" => Ok(StreamContent::Frame),
            "IMUS" => Ok(StreamContent::Imus),
            "TRIG" => Ok(StreamContent::Triggers),
            _ => Err(ParseError::new("unsupported stream type")),
        }
    }
}

impl std::fmt::Display for StreamContent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            formatter,
            "{}",
            match self {
                StreamContent::Events => "EVTS",
                StreamContent::Frame => "FRME",
                StreamContent::Imus => "IMUS",
                StreamContent::Triggers => "TRIG",
            }
        )
    }
}

pub struct Stream {
    pub content: StreamContent,
    pub width: u16,
    pub height: u16,
}

pub struct Decoder {
    pub id_to_stream: std::collections::HashMap<u32, Stream>,
    file: std::fs::File,
    position: i64,
    compression: ioheader_generated::Compression,
    file_data_position: i64,
}

impl Decoder {
    pub fn new<P: std::convert::AsRef<std::path::Path>>(path: P) -> Result<Self, ParseError> {
        let mut decoder = Decoder {
            id_to_stream: std::collections::HashMap::new(),
            file: std::fs::File::open(path)?,
            position: 0i64,
            file_data_position: 0,
            compression: ioheader_generated::Compression::None,
        };
        {
            let mut magic_number_buffer = [0; MAGIC_NUMBER.len()];
            decoder.file.read_exact(&mut magic_number_buffer)?;
            if std::str::from_utf8(&magic_number_buffer)? != MAGIC_NUMBER {
                return Err(ParseError::new(
                    "the file does not contain AEDAT4 data (wrong magic number)",
                ));
            }
            decoder.position += MAGIC_NUMBER.len() as i64;
        }
        let length = {
            let mut bytes = [0; 4];
            decoder.file.read_exact(&mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        decoder.position += 4i64 + length as i64;
        {
            let mut buffer = std::vec![0; length as usize];
            decoder.file.read_exact(&mut buffer)?;
            let ioheader = unsafe { ioheader_generated::root_as_ioheader_unchecked(&buffer) };
            decoder.compression = ioheader.compression();
            decoder.file_data_position = ioheader.file_data_position();
            let description = match ioheader.description() {
                Some(content) => content,
                None => return Err(ParseError::new("the description is empty")),
            };
            let document = roxmltree::Document::parse(description)?;
            let dv_node = match document.root().first_child() {
                Some(content) => content,
                None => return Err(ParseError::new("the description has no dv node")),
            };
            if !dv_node.has_tag_name("dv") {
                return Err(ParseError::new("unexpected dv node tag"));
            }
            let output_node = match dv_node.children().find(|node| {
                node.is_element()
                    && node.has_tag_name("node")
                    && node.attribute("name") == Some("outInfo")
            }) {
                Some(content) => content,
                None => return Err(ParseError::new("the description has no output node")),
            };
            for stream_node in output_node.children() {
                if stream_node.is_element() && stream_node.has_tag_name("node") {
                    if !stream_node.has_tag_name("node") {
                        return Err(ParseError::new("unexpected stream node tag"));
                    }
                    let stream_id = match stream_node.attribute("name") {
                        Some(content) => content,
                        None => return Err(ParseError::new("missing stream node id")),
                    }
                    .parse::<u32>()?;
                    let identifier = match stream_node.children().find(|node| {
                        node.is_element()
                            && node.has_tag_name("attr")
                            && node.attribute("key") == Some("typeIdentifier")
                    }) {
                        Some(content) => match content.text() {
                            Some(content) => content,
                            None => {
                                return Err(ParseError::new("empty stream node type identifier"))
                            }
                        },
                        None => return Err(ParseError::new("missing stream node type identifier")),
                    }
                    .to_string();
                    let mut width = 0u16;
                    let mut height = 0u16;
                    if identifier == "EVTS" || identifier == "FRME" {
                        let info_node = match stream_node.children().find(|node| {
                            node.is_element()
                                && node.has_tag_name("node")
                                && node.attribute("name") == Some("info")
                        }) {
                            Some(content) => content,
                            None => return Err(ParseError::new("missing info node")),
                        };
                        width = match info_node.children().find(|node| {
                            node.is_element()
                                && node.has_tag_name("attr")
                                && node.attribute("key") == Some("sizeX")
                        }) {
                            Some(content) => match content.text() {
                                Some(content) => content,
                                None => return Err(ParseError::new("empty sizeX attribute")),
                            },
                            None => return Err(ParseError::new("missing sizeX attribute")),
                        }
                        .parse::<u16>()?;
                        height = match info_node.children().find(|node| {
                            node.is_element()
                                && node.has_tag_name("attr")
                                && node.attribute("key") == Some("sizeY")
                        }) {
                            Some(content) => match content.text() {
                                Some(content) => content,
                                None => return Err(ParseError::new("empty sizeX attribute")),
                            },
                            None => return Err(ParseError::new("missing sizeX attribute")),
                        }
                        .parse::<u16>()?;
                    }
                    if decoder
                        .id_to_stream
                        .insert(
                            stream_id,
                            Stream {
                                content: StreamContent::from(&identifier)?,
                                width,
                                height,
                            },
                        )
                        .is_some()
                    {
                        return Err(ParseError::new("duplicated stream id"));
                    }
                }
            }
        }
        if decoder.id_to_stream.is_empty() {
            return Err(ParseError::new("no stream found in the description"));
        }
        Ok(decoder)
    }
}

pub struct Packet {
    pub buffer: std::vec::Vec<u8>,
    pub stream_id: u32,
}

impl Iterator for Decoder {
    type Item = Result<Packet, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.file_data_position > -1 && self.position == self.file_data_position {
            return None;
        }
        let mut packet = Packet {
            buffer: Vec::new(),
            stream_id: {
                let mut bytes = [0; 4];
                match self.file.read_exact(&mut bytes) {
                    Ok(()) => (),
                    Err(_) => return None,
                }
                u32::from_le_bytes(bytes)
            },
        };
        let length = {
            let mut bytes = [0; 4];
            if let Err(error) = self.file.read_exact(&mut bytes) {
                return Some(Err(ParseError::from(error)));
            }
            u32::from_le_bytes(bytes)
        };
        self.position += 8i64 + length as i64;
        let mut raw_buffer = std::vec![0; length as usize];
        if let Err(error) = self.file.read_exact(&mut raw_buffer) {
            return Some(Err(ParseError::from(error)));
        }
        match self.compression {
            ioheader_generated::Compression::None => {
                std::mem::swap(&mut raw_buffer, &mut packet.buffer)
            }
            ioheader_generated::Compression::Lz4 | ioheader_generated::Compression::Lz4High => {
                match lz4::Decoder::new(&raw_buffer[..]) {
                    Ok(mut result) => {
                        if let Err(error) = result.read_to_end(&mut packet.buffer) {
                            return Some(Err(ParseError::from(error)));
                        }
                    }
                    Err(error) => return Some(Err(ParseError::from(error))),
                }
            }
            ioheader_generated::Compression::Zstd | ioheader_generated::Compression::ZstdHigh => {
                match zstd::stream::Decoder::new(&raw_buffer[..]) {
                    Ok(mut result) => {
                        if let Err(error) = result.read_to_end(&mut packet.buffer) {
                            return Some(Err(ParseError::from(error)));
                        }
                    }
                    Err(error) => return Some(Err(ParseError::from(error))),
                }
            }
            _ => return Some(Err(ParseError::new("unknown compression algorithm"))),
        }
        let expected_content = &(match self.id_to_stream.get(&packet.stream_id) {
            Some(content) => content,
            None => return Some(Err(ParseError::new("unknown stream id"))),
        }
        .content);
        if !flatbuffers::buffer_has_identifier(&packet.buffer, &expected_content.to_string(), true)
        {
            return Some(Err(ParseError::new(
                "the stream id and the identifier do not match",
            )));
        }
        Some(Ok(packet))
    }
}
