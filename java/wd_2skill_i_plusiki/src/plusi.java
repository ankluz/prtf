import java.util.Objects;

public class plusi {
    private int odin;
    private float dva;
    private String tri;
    public int chetire;

    public void setOdin(int odin) {
        this.odin = odin;
    }

    public void setDva(float dva) {
        this.dva = dva;
    }

    public void setTri(String tri) {
        this.tri = tri;
    }

    public void setChetire(int chetire) {
        this.chetire = chetire;
    }

    public float getDva() {
        return dva;
    }

    public String getTri() {
        return tri;
    }

    @Override
    public String toString() {
        return "plusi{" +
                "odin=" + odin +
                ", dva=" + dva +
                ", tri='" + tri + '\'' +
                ", chetire=" + chetire +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        plusi plusi = (plusi) o;
        return odin == plusi.odin && Float.compare(plusi.dva, dva) == 0 && chetire == plusi.chetire && Objects.equals(tri, plusi.tri);
    }

    @Override
    public int hashCode() {
        return Objects.hash(odin, dva, tri, chetire);
    }

    public int getChetire() {
        return chetire;
    }

    public int getOdin() {
        return odin;
    }
}
