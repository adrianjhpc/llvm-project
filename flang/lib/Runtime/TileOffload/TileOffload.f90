! Public TileOffload device-selection interfaces. Compile once with the host compiler.
module TileOffload
  use, intrinsic :: iso_c_binding, only: c_int32_t
  implicit none
  private
  public :: tileoff_get_num_devices, tileoff_set_device_num, tileoff_get_device_num
  interface
    function tileoff_get_num_devices() bind(C, name="tileoff_get_num_devices") result(n)
      import :: c_int32_t
      integer(c_int32_t) :: n
    end function
    subroutine tileoff_set_device_num(device) bind(C, name="tileoff_set_device_num")
      import :: c_int32_t
      integer(c_int32_t), value :: device
    end subroutine
    function tileoff_get_device_num() bind(C, name="tileoff_get_device_num") result(device)
      import :: c_int32_t
      integer(c_int32_t) :: device
    end function
  end interface
end module
