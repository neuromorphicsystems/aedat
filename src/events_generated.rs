// automatically generated by the FlatBuffers compiler, do not modify


// @generated

use core::mem;
use core::cmp::Ordering;

extern crate flatbuffers;
use self::flatbuffers::{EndianScalar, Follow};

// struct Event, aligned to 8
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq)]
pub struct Event(pub [u8; 16]);
impl Default for Event { 
  fn default() -> Self { 
    Self([0; 16])
  }
}
impl core::fmt::Debug for Event {
  fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
    f.debug_struct("Event")
      .field("t", &self.t())
      .field("x", &self.x())
      .field("y", &self.y())
      .field("on", &self.on())
      .finish()
  }
}

impl flatbuffers::SimpleToVerifyInSlice for Event {}
impl<'a> flatbuffers::Follow<'a> for Event {
  type Inner = &'a Event;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    <&'a Event>::follow(buf, loc)
  }
}
impl<'a> flatbuffers::Follow<'a> for &'a Event {
  type Inner = &'a Event;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    flatbuffers::follow_cast_ref::<Event>(buf, loc)
  }
}
impl<'b> flatbuffers::Push for Event {
    type Output = Event;
    #[inline]
    unsafe fn push(&self, dst: &mut [u8], _written_len: usize) {
        let src = ::core::slice::from_raw_parts(self as *const Event as *const u8, Self::size());
        dst.copy_from_slice(src);
    }
}

impl<'a> flatbuffers::Verifiable for Event {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.in_buffer::<Self>(pos)
  }
}

impl<'a> Event {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    t: i64,
    x: i16,
    y: i16,
    on: bool,
  ) -> Self {
    let mut s = Self([0; 16]);
    s.set_t(t);
    s.set_x(x);
    s.set_y(y);
    s.set_on(on);
    s
  }

  pub fn t(&self) -> i64 {
    let mut mem = core::mem::MaybeUninit::<<i64 as EndianScalar>::Scalar>::uninit();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    EndianScalar::from_little_endian(unsafe {
      core::ptr::copy_nonoverlapping(
        self.0[0..].as_ptr(),
        mem.as_mut_ptr() as *mut u8,
        core::mem::size_of::<<i64 as EndianScalar>::Scalar>(),
      );
      mem.assume_init()
    })
  }

  pub fn set_t(&mut self, x: i64) {
    let x_le = x.to_little_endian();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    unsafe {
      core::ptr::copy_nonoverlapping(
        &x_le as *const _ as *const u8,
        self.0[0..].as_mut_ptr(),
        core::mem::size_of::<<i64 as EndianScalar>::Scalar>(),
      );
    }
  }

  pub fn x(&self) -> i16 {
    let mut mem = core::mem::MaybeUninit::<<i16 as EndianScalar>::Scalar>::uninit();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    EndianScalar::from_little_endian(unsafe {
      core::ptr::copy_nonoverlapping(
        self.0[8..].as_ptr(),
        mem.as_mut_ptr() as *mut u8,
        core::mem::size_of::<<i16 as EndianScalar>::Scalar>(),
      );
      mem.assume_init()
    })
  }

  pub fn set_x(&mut self, x: i16) {
    let x_le = x.to_little_endian();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    unsafe {
      core::ptr::copy_nonoverlapping(
        &x_le as *const _ as *const u8,
        self.0[8..].as_mut_ptr(),
        core::mem::size_of::<<i16 as EndianScalar>::Scalar>(),
      );
    }
  }

  pub fn y(&self) -> i16 {
    let mut mem = core::mem::MaybeUninit::<<i16 as EndianScalar>::Scalar>::uninit();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    EndianScalar::from_little_endian(unsafe {
      core::ptr::copy_nonoverlapping(
        self.0[10..].as_ptr(),
        mem.as_mut_ptr() as *mut u8,
        core::mem::size_of::<<i16 as EndianScalar>::Scalar>(),
      );
      mem.assume_init()
    })
  }

  pub fn set_y(&mut self, x: i16) {
    let x_le = x.to_little_endian();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    unsafe {
      core::ptr::copy_nonoverlapping(
        &x_le as *const _ as *const u8,
        self.0[10..].as_mut_ptr(),
        core::mem::size_of::<<i16 as EndianScalar>::Scalar>(),
      );
    }
  }

  pub fn on(&self) -> bool {
    let mut mem = core::mem::MaybeUninit::<<bool as EndianScalar>::Scalar>::uninit();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    EndianScalar::from_little_endian(unsafe {
      core::ptr::copy_nonoverlapping(
        self.0[12..].as_ptr(),
        mem.as_mut_ptr() as *mut u8,
        core::mem::size_of::<<bool as EndianScalar>::Scalar>(),
      );
      mem.assume_init()
    })
  }

  pub fn set_on(&mut self, x: bool) {
    let x_le = x.to_little_endian();
    // Safety:
    // Created from a valid Table for this object
    // Which contains a valid value in this slot
    unsafe {
      core::ptr::copy_nonoverlapping(
        &x_le as *const _ as *const u8,
        self.0[12..].as_mut_ptr(),
        core::mem::size_of::<<bool as EndianScalar>::Scalar>(),
      );
    }
  }

}

pub enum EventPacketOffset {}
#[derive(Copy, Clone, PartialEq)]

pub struct EventPacket<'a> {
  pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for EventPacket<'a> {
  type Inner = EventPacket<'a>;
  #[inline]
  unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
    Self { _tab: flatbuffers::Table::new(buf, loc) }
  }
}

impl<'a> EventPacket<'a> {
  pub const VT_ELEMENTS: flatbuffers::VOffsetT = 4;

  #[inline]
  pub unsafe fn init_from_table(table: flatbuffers::Table<'a>) -> Self {
    EventPacket { _tab: table }
  }
  #[allow(unused_mut)]
  pub fn create<'bldr: 'args, 'args: 'mut_bldr, 'mut_bldr, A: flatbuffers::Allocator + 'bldr>(
    _fbb: &'mut_bldr mut flatbuffers::FlatBufferBuilder<'bldr, A>,
    args: &'args EventPacketArgs<'args>
  ) -> flatbuffers::WIPOffset<EventPacket<'bldr>> {
    let mut builder = EventPacketBuilder::new(_fbb);
    if let Some(x) = args.elements { builder.add_elements(x); }
    builder.finish()
  }


  #[inline]
  pub fn elements(&self) -> Option<flatbuffers::Vector<'a, Event>> {
    // Safety:
    // Created from valid Table for this object
    // which contains a valid value in this slot
    unsafe { self._tab.get::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'a, Event>>>(EventPacket::VT_ELEMENTS, None)}
  }
}

impl flatbuffers::Verifiable for EventPacket<'_> {
  #[inline]
  fn run_verifier(
    v: &mut flatbuffers::Verifier, pos: usize
  ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
    use self::flatbuffers::Verifiable;
    v.visit_table(pos)?
     .visit_field::<flatbuffers::ForwardsUOffset<flatbuffers::Vector<'_, Event>>>("elements", Self::VT_ELEMENTS, false)?
     .finish();
    Ok(())
  }
}
pub struct EventPacketArgs<'a> {
    pub elements: Option<flatbuffers::WIPOffset<flatbuffers::Vector<'a, Event>>>,
}
impl<'a> Default for EventPacketArgs<'a> {
  #[inline]
  fn default() -> Self {
    EventPacketArgs {
      elements: None,
    }
  }
}

pub struct EventPacketBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
  fbb_: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
  start_: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}
impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> EventPacketBuilder<'a, 'b, A> {
  #[inline]
  pub fn add_elements(&mut self, elements: flatbuffers::WIPOffset<flatbuffers::Vector<'b , Event>>) {
    self.fbb_.push_slot_always::<flatbuffers::WIPOffset<_>>(EventPacket::VT_ELEMENTS, elements);
  }
  #[inline]
  pub fn new(_fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> EventPacketBuilder<'a, 'b, A> {
    let start = _fbb.start_table();
    EventPacketBuilder {
      fbb_: _fbb,
      start_: start,
    }
  }
  #[inline]
  pub fn finish(self) -> flatbuffers::WIPOffset<EventPacket<'a>> {
    let o = self.fbb_.end_table(self.start_);
    flatbuffers::WIPOffset::new(o.value())
  }
}

impl core::fmt::Debug for EventPacket<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("EventPacket");
      ds.field("elements", &self.elements());
      ds.finish()
  }
}
#[inline]
/// Verifies that a buffer of bytes contains a `EventPacket`
/// and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_event_packet_unchecked`.
pub fn root_as_event_packet(buf: &[u8]) -> Result<EventPacket, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root::<EventPacket>(buf)
}
#[inline]
/// Verifies that a buffer of bytes contains a size prefixed
/// `EventPacket` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `size_prefixed_root_as_event_packet_unchecked`.
pub fn size_prefixed_root_as_event_packet(buf: &[u8]) -> Result<EventPacket, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root::<EventPacket>(buf)
}
#[inline]
/// Verifies, with the given options, that a buffer of bytes
/// contains a `EventPacket` and returns it.
/// Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_event_packet_unchecked`.
pub fn root_as_event_packet_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<EventPacket<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::root_with_opts::<EventPacket<'b>>(opts, buf)
}
#[inline]
/// Verifies, with the given verifier options, that a buffer of
/// bytes contains a size prefixed `EventPacket` and returns
/// it. Note that verification is still experimental and may not
/// catch every error, or be maximally performant. For the
/// previous, unchecked, behavior use
/// `root_as_event_packet_unchecked`.
pub fn size_prefixed_root_as_event_packet_with_opts<'b, 'o>(
  opts: &'o flatbuffers::VerifierOptions,
  buf: &'b [u8],
) -> Result<EventPacket<'b>, flatbuffers::InvalidFlatbuffer> {
  flatbuffers::size_prefixed_root_with_opts::<EventPacket<'b>>(opts, buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a EventPacket and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid `EventPacket`.
pub unsafe fn root_as_event_packet_unchecked(buf: &[u8]) -> EventPacket {
  flatbuffers::root_unchecked::<EventPacket>(buf)
}
#[inline]
/// Assumes, without verification, that a buffer of bytes contains a size prefixed EventPacket and returns it.
/// # Safety
/// Callers must trust the given bytes do indeed contain a valid size prefixed `EventPacket`.
pub unsafe fn size_prefixed_root_as_event_packet_unchecked(buf: &[u8]) -> EventPacket {
  flatbuffers::size_prefixed_root_unchecked::<EventPacket>(buf)
}
pub const EVENT_PACKET_IDENTIFIER: &str = "EVTS";

#[inline]
pub fn event_packet_buffer_has_identifier(buf: &[u8]) -> bool {
  flatbuffers::buffer_has_identifier(buf, EVENT_PACKET_IDENTIFIER, false)
}

#[inline]
pub fn event_packet_size_prefixed_buffer_has_identifier(buf: &[u8]) -> bool {
  flatbuffers::buffer_has_identifier(buf, EVENT_PACKET_IDENTIFIER, true)
}

#[inline]
pub fn finish_event_packet_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
    root: flatbuffers::WIPOffset<EventPacket<'a>>) {
  fbb.finish(root, Some(EVENT_PACKET_IDENTIFIER));
}

#[inline]
pub fn finish_size_prefixed_event_packet_buffer<'a, 'b, A: flatbuffers::Allocator + 'a>(fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>, root: flatbuffers::WIPOffset<EventPacket<'a>>) {
  fbb.finish_size_prefixed(root, Some(EVENT_PACKET_IDENTIFIER));
}
