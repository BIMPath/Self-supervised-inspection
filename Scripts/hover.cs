using UnityEngine;

public class BridgePart : MonoBehaviour
{
    public Material normalMaterial;
    public Material highlightMaterial;
    private Renderer rend;
 
    public string imageName; // reference to your image
 
    void Start()
    {
        rend = GetComponent<Renderer>();
        rend.material = normalMaterial;
    }
 
    public void OnHoverEnter()
    {
        rend.material = highlightMaterial;
    }
 
    public void OnHoverExit()
    {
        rend.material = normalMaterial;
    }
 
    public void OnClick()
    {
        UIManager.Instance.ShowPanel(imageName);
    }
}
