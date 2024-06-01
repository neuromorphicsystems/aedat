// automatically generated by the FlatBuffers compiler, do not modify


// @generated

use core::mem;
use core::cmp::Ordering;

extern crate flatbuffers;
use self::flatbuffers::{EndianScalar, Follow};

#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MIN_TRIGGER_SOURCE: i8 = 0;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
pub const ENUM_MAX_TRIGGER_SOURCE: i8 = 9;
#[deprecated(since = "2.0.0", note = "Use associated constants instead. This will no longer be generated in 2021.")]
#[allow(non_camel_case_types)]
pub const ENUM_VALUES_TRIGGER_SOURCE: [TriggerSource; 10] = [
  TriggerSource::TimestampReset,
  TriggerSource::ExternalSignalRisingEdge,
  TriggerSource::ExternalSignalFallingEdge,
  TriggerSource::ExternalSignalPulse,
  TriggerSource::ExternalGeneratorRisingEdge,
  TriggerSource::ExternalGeneratorFallingEdge,
  TriggerSource::FrameBegin,
  TriggerSource::FrameEnd,
  TriggerSource::ExposureBegin,
  TriggerSource::ExposureEnd,
];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct TriggerSource(pub i8);
#[allow(non_upper_case_globals)]
impl TriggerSource {
  pub const TimestampReset: Self = Self(0);
  pub const ExternalSignalRisingEdge: Self = Self(1);
  pub const ExternalSignalFallingEdge: Self = Self(2);
  pub const ExternalSignalPulse: Self = Self(3);
  pub const ExternalGeneratorRisingEdge: Self = Self(4);
  pub const ExternalGeneratorFallingEdge: Self = Self(5);
  pub const FrameBegin: Self = Self(6);
  pub const FrameEnd: Self = Self(7);
  pub const ExposureBegin: Self = Self(8);
  pub const ExposureEnd: Self = Self(9);

  pub const ENUM_MIN: i8 = 0;
  pub const ENUM_MAX: i8 = 9;
  pub const ENUM_VALUES: &'static [Self] = &[
    Self::TimestampReset,
    Self::ExternalSignalRisingEdge,
    Self::ExternalSignalFallingEdge,
    Self::ExternalSignalPulse,
    Self::ExternalGeneratorRisingEdge,
    Self::ExternalGeneratorFallingEdge,
    Self::FrameBegin,
    Self::FrameEnd,
    Self::ExposureBegin,
    Self::ExposureEnd,
  ];
  /// Returns the variant's name or "" if unknown.
  pub fn variant_name(self) -> Option<&'static str> {
    match self {
      Self::TimestampReset => Some("TimestampReset"),
      Self::ExternalSignalRisingEdge => Some("ExternalSignalRisingEdge"),
      Self::ExternalSignalFallingEdge => Some("ExternalSignalFallingEdge"),
      Self::ExternalSignalPulse => Some("ExternalSignalPulse"),
      Self::ExternalGeneratorRisingEdge => Some("ExternalGeneratorRisingEdge"),
      Self::ExternalGeneratorFallingEdge => Some("ExternalGeneratorFallingEdge"),
      Self::FrameBegin => Some("FrameBegin"),
      Self::FrameEnd => Some("FrameEnd"),
      Self::ExposureBegin => Some("ExposureBegin"),
      Self::ExposureEnd => Some("ExposureEnd"),
      _ => None,
    }
  }
}
impl core::fmt::Debug for TriggerSource {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    if let Some(name) = self.variant_name() {
      f.write_str(name)
    } else {
      f.write_fmt(format_args!("<UNKNOWN {:?}>", self.0))
    }
  }
}
impl<'a> flatbuffers::Follow<'a> for TriggerSource {
  type Inner = Self;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    let b = flatbuffers::read_scalar_at::<i8>(buf, loc);
    Self(b)
  }
}

impl flatbuffers::Push for TriggerSource {
    type Output = TriggerSource;
    #[inline]
    unsafe fn push(&self, dst: &mut [u8], _written_len: usize) {
        flatbuffers::emplace_scalar::<i8>(dst, self.0);
    }
}

impl flatbuffers::EndianScalar for TriggerSource {
  type Scalar = i8;
  #[inline]
  fn to_little_endian(self) -> i8 {
    self.0.to_le()
  }
  #[inline]
  #[allow(clippy::wrong_self_convention)]
  fn from_little_endian(v: i8) -> Self {
    let b = i8::from_le(v);
    Self(b)
  }
}

impl<'a> flatbuffers::Verifiable for TriggerSource {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    i8::run_verifier(v, pos)
  }
}

impl flatbuffers::SimpleToVerifyInSlice for TriggerSource {}
pub enum TriggerOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct Trigger<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for Trigger<'a> {
  type Inner = Trigger<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> Trigger<'a> {
  pub const VT_T: flatbuffers::VOffsetT = 4;
  pub const VT_SOURCE: flatbuffers::VOffsetT = 6;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    Trigger { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args TriggerArgs
  ) -> flatbuffers::WIPOffset<Trigger<'bldr>> {
    let mut builder = TriggerBuilder::new(_fbb);
    builder.add_t(args.t);
    builder.add_source(args.source);
    builder.finish()
  }


  #[inline]
  pub fn t(&self) -> i64 {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<i64>(Trigger::VT_T, Some(0)).unwrap()}
  }
  #[inline]
  pub fn source(&self) -> TriggerSource {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<TriggerSource>(Trigger::VT_SOURCE, Some(TriggerSource::TimestampReset)).unwrap()}
  }
}

impl flatbuffers::Verifiable for Trigger<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<i64>("t", Self::VT_T, false)?
     .visit_field::<TriggerSource>("source", Self::VT_SOURCE, false)?
     .finish();
    Ok(())
  }
}
pub struct TriggerArgs {
    pub t: i64,
    pub source: TriggerSource,
}
impl<'a> Default for TriggerArgs {
  #[inline]
  fn default() -> Self {
    TriggerArgs {
      t: 0,
      source: TriggerSource::TimestampReset,
    }
  }
}

pub struct TriggerBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> TriggerBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_t(&mut self, t: i64) {
    self.fbb_.push_slot::<i64>(Trigger::VT_T, t, 0);
  }
  #[inline]
  pub fn add_source(&mut self, source: TriggerSource) {
    self.fbb_.push_slot::<TriggerSource>(Trigger::VT_SOURCE, source, TriggerSource::TimestampReset);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> TriggerBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    TriggerBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<Trigger<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for Trigger<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("Trigger");
      ds.field("t", &self.t());
      ds.field("source", &self.source());
      ds.finish()
  }
}
pub enum TriggerPacketOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct TriggerPacket<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for TriggerPacket<'a> {
  type Inner = TriggerPacket<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> TriggerPacket<'a> {
  pub const VT_ELEMENTS: flatbuffers::VOffsetT = 4;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    TriggerPacket { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args TriggerPacketArgs<'args>
  ) -> flatbuffers::WIPOffset<TriggerPacket<'bldr>> {
    let mut builder = TriggerPacketBuilder::new(_fbb);
    if let Some(x) = args.elements { builder.add_elements(x); }
    builder.finish()
  }


  #[inline]
  pub fn elements(&self) -> Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Trigger<'a>>>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Trigger>>>>(TriggerPacket::VT_ELEMENTS, None)}
  }
}

impl flatbuffers::Verifiable for TriggerPacket<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<Trigger>>>>("elements", Self::VT_ELEMENTS, false)?
     .finish();
    Ok(())
  }
}
pub struct TriggerPacketArgs<'a> {
    pub elements: Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Trigger<'a>>>>>,
}
impl<'a> Default for TriggerPacketArgs<'a> {
  #[inline]
  fn default() -> Self {
    TriggerPacketArgs {
      elements: None,
    }
  }
}

pub struct TriggerPacketBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> TriggerPacketBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_elements(&mut self, elements: flatbuffers::WIPOffset<flatbuffers::Vector<'b , flatbuffers::ForwardsUOffset<Trigger<'b >>>>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(TriggerPacket::VT_ELEMENTS, elements);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> TriggerPacketBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    TriggerPacketBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<TriggerPacket<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for TriggerPacket<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("TriggerPacket");
      ds.field("elements", &self.elements());
      ds.finish()
  }
}
#[inline]
/// Verifies that a buffer of bytes contains a `TriggerPacket`
/// and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_trigger_packet_unchecked`.
pub fn root_as_trigger_packet(buf: &[u8]) -> Result<TriggerPacket, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root::<TriggerPacket>(buf)
}
#[inline]
/// Verifies that a buffer of bytes contains a size prefixed
/// `TriggerPacket` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `size_prefixed_root_as_trigger_packet_unchecked`.
pub fn size_prefixed_root_as_trigger_packet(buf: &[u8]) -> Result<TriggerPacket, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root::<TriggerPacket>(buf)
}
#[inline]
/// Verifies, with the given options, that a buffer of bytes
/// contains a `TriggerPacket` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_trigger_packet_unchecked`.
pub fn root_as_trigger_packet_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<TriggerPacket<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root_with_opts::<TriggerPacket<'b>>(opts, buf)
}
#[inline]
/// Verifies, with the given verifier options, that a buffer of
/// bytes contains a size prefixed `TriggerPacket` and returns
/// it. Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_trigger_packet_unchecked`.
pub fn size_prefixed_root_as_trigger_packet_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<TriggerPacket<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root_with_opts::<TriggerPacket<'b>>(opts, buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a TriggerPacket and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid `TriggerPacket`.
pub unsafe fn root_as_trigger_packet_unchecked(buf: &[u8]) -> TriggerPacket {
  flatbuffers::root_unchecked::<TriggerPacket>(buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a size prefixed TriggerPacket and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid size prefixed `TriggerPacket`.
pub unsafe fn size_prefixed_root_as_trigger_packet_unchecked(buf: &[u8]) -> TriggerPacket {
  flatbuffers::size_prefixed_root_unchecked::<TriggerPacket>(buf)
}
pub const TRIGGER_PACKET_IDENTIFIER: &str = "TRIG";

#[inline]
pub fn trigger_packet_buffer_has_identifier(buf: &[u8]) -> bool {
  flatbuffers::buffer_has_identifier(buf, TRIGGER_PACKET_IDENTIFIER, false)
}

#[inline]
pub fn trigger_packet_size_prefixed_buffer_has_identifier(buf: &[u8]) -> bool {
  flatbuffers::buffer_has_identifier(buf, TRIGGER_PACKET_IDENTIFIER, true)
}

#[inline]
pub fn finish_trigger_packet_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
    root: flatbuffers::WIPOffset<TriggerPacket<'a>>) {
  fbb.finish(root, Some(TRIGGER_PACKET_IDENTIFIER));
}

#[inline]
pub fn finish_size_prefixed_trigger_packet_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>, root: flatbuffers::WIPOffset<TriggerPacket<'a>>) {
  fbb.finish_size_prefixed(root, Some(TRIGGER_PACKET_IDENTIFIER));
}
